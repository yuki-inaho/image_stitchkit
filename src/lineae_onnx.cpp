#include "internal.hpp"
#include "stitchkit/lineae.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <set>

namespace stitchkit::detail {
namespace {
class OnnxLineae final : public IStructureDetector {
    Ort::Env environment_{ORT_LOGGING_LEVEL_WARNING, "stitchkit-lineae"};
    mutable Ort::Session session_{nullptr};
    Size model_size_;
    std::int64_t count_ = 0;
    std::string variant_, device_ = "cpu";
    std::unique_ptr<IStructureDetector> edges_ = make_edge_detector();

    void validate_schema() {
        if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 2)
            throw Error("Unexpected LINEAE input/output count");
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session_.GetInputNameAllocated(0, allocator);
        if (std::string(input_name.get()) != "images")
            throw Error("LINEAE input must be named images");
        const auto input_type = session_.GetInputTypeInfo(0);
        const auto input_tensor = input_type.GetTensorTypeAndShapeInfo();
        const auto input = input_tensor.GetShape();
        if (input_tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input.size() != 4 || input[0] != 1 || input[1] != 3 || input[2] < 2 || input[3] < 2 ||
            input[2] > 4096 || input[3] > 4096)
            throw Error("LINEAE expects fixed float32 input [1,3,H,W]");
        model_size_ = {static_cast<int>(input[3]), static_cast<int>(input[2])};
        std::set<std::string> names;
        for (std::size_t k = 0; k < 2; ++k) {
            const auto name = session_.GetOutputNameAllocated(k, allocator);
            const std::string text = name.get();
            names.insert(text);
            const auto type = session_.GetOutputTypeInfo(k);
            const auto tensor = type.GetTensorTypeAndShapeInfo();
            const auto shape = tensor.GetShape();
            const int dimensions = text == "pred_logits" ? 2 : text == "pred_lines" ? 4 : 0;
            if (!dimensions || tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[1] > 100000 ||
                shape[2] != dimensions)
                throw Error("Invalid LINEAE output schema");
            if (count_ && count_ != shape[1])
                throw Error("LINEAE outputs disagree on K");
            count_ = shape[1];
        }
        if (names != std::set<std::string>{"pred_logits", "pred_lines"})
            throw Error("LINEAE output names do not match");
    }

  public:
    explicit OnnxLineae(const Options& o) : variant_(o.lineae_variant) {
        (void)lineae_normalization(variant_);
        if (!std::filesystem::is_regular_file(o.lineae_model))
            throw Error("LINEAE model file does not exist");
        try {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (o.lineae_device == Device::cuda) {
                const auto providers = Ort::GetAvailableProviders();
                if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") ==
                    providers.end())
                    throw Error("LINEAE explicitly requested CUDA, but the CUDA Execution Provider "
                                "is not available");
                OrtCUDAProviderOptions cuda_options{};
                cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                // Reject unsupported nodes instead of silently executing them on the CPU.
                session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
                device_ = "cuda";
            }
            // LINEAE auto intentionally uses CPU for portability; renderer auto is independent.
            session_ = Ort::Session(environment_, o.lineae_model.c_str(), session_options);
            validate_schema();
        } catch (const Ort::Exception& e) {
            throw Error(std::string("LINEAE initialization failed: ") + e.what());
        }
    }
    std::string device_name() const override { return device_; }
    std::vector<Structure> detect(const Image& image, const Options& o) const override {
        try {
            auto data = prepare_lineae_input(image, model_size_, variant_);
            const std::array<std::int64_t, 4> shape{1, 3, model_size_.height, model_size_.width};
            const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input = Ort::Value::CreateTensor<float>(memory, data.data(), data.size(),
                                                         shape.data(), shape.size());
            const char* input_names[] = {"images"};
            const char* output_names[] = {"pred_logits", "pred_lines"};
            // ONNX Runtime supports concurrent Run calls; all input/output buffers are local.
            auto output =
                session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 2);
            for (int k = 0; k < 2; ++k) {
                if (!output[k].IsTensor())
                    throw Error("LINEAE returned a non-tensor output");
                const auto info = output[k].GetTensorTypeAndShapeInfo();
                if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                    info.GetShape() != std::vector<std::int64_t>{1, count_, k ? 4 : 2})
                    throw Error("LINEAE runtime output shape differs from its schema");
            }
            const auto* scores = output[0].GetTensorData<float>();
            const auto* lines = output[1].GetTensorData<float>();
            const int line_limit = std::max(1, o.max_structures / 2);
            auto result = decode_lineae_output(std::vector<float>(scores, scores + 2 * count_),
                                               std::vector<float>(lines, lines + 4 * count_),
                                               image.size(), o.lineae_threshold, line_limit);
            auto contours = edges_->detect(image, o);
            for (auto& contour : contours) {
                if (result.size() >= static_cast<std::size_t>(o.max_structures))
                    break;
                result.push_back(std::move(contour));
            }
            return result;
        } catch (const Ort::Exception& e) {
            throw Error(std::string("LINEAE inference failed: ") + e.what());
        }
    }
};
} // namespace
std::unique_ptr<IStructureDetector> make_onnx_lineae(const Options& o) {
    return std::make_unique<OnnxLineae>(o);
}
} // namespace stitchkit::detail
