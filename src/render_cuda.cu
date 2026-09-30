#include "stitchkit/interfaces.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <string>

namespace stitchkit::detail {
void validate_plan(const std::vector<Image>& images, const WarpPlan& plan, const Options& options);
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw Error(std::string(operation) + ": " + cudaGetErrorString(status));
}
template <class T> class DeviceBuffer {
    T* pointer_ = nullptr;

  public:
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)),
              "CUDA allocation");
    }
    ~DeviceBuffer() {
        if (pointer_)
            (void)cudaFree(pointer_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return pointer_; }
    void upload(const T* data, std::size_t count) {
        check(cudaMemcpy(pointer_, data, count * sizeof(T), cudaMemcpyHostToDevice), "CUDA upload");
    }
    void download(T* data, std::size_t count) const {
        check(cudaMemcpy(data, pointer_, count * sizeof(T), cudaMemcpyDeviceToHost),
              "CUDA download");
    }
};
__global__ void accumulate(const unsigned char* image, int width, int height, const float* map_x,
                           const float* map_y, std::size_t count, float feather, float4* sums) {
    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    const float x = map_x[p], y = map_y[p];
    if (!isfinite(x) || !isfinite(y) || x < 0 || y < 0 || x > width - 1 || y > height - 1)
        return;
    const int x0 = int(x), y0 = int(y), x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    const float u = x - x0, v = y - y0;
    const float distance = fminf(fminf(x + 1, y + 1), fminf(width - x, height - y));
    const float weight = fminf(1.f, distance / feather);
    float rgb[3];
    for (int c = 0; c < 3; ++c) {
        const float a = image[(static_cast<std::size_t>(y0) * width + x0) * 3 + c],
                    b = image[(static_cast<std::size_t>(y0) * width + x1) * 3 + c];
        const float d = image[(static_cast<std::size_t>(y1) * width + x0) * 3 + c],
                    e = image[(static_cast<std::size_t>(y1) * width + x1) * 3 + c];
        rgb[c] = ((a * (1 - u) + b * u) * (1 - v) + (d * (1 - u) + e * u) * v) * weight;
    }
    float4 value = sums[p];
    value.x += rgb[0];
    value.y += rgb[1];
    value.z += rgb[2];
    value.w += weight;
    sums[p] = value;
}
__global__ void finish(const float4* sums, std::size_t count, unsigned char* rgb,
                       unsigned char* mask) {
    const std::size_t p = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= count)
        return;
    const float4 value = sums[p];
    mask[p] = value.w > 0 ? 255 : 0;
    const float colors[3] = {value.x, value.y, value.z};
    for (int c = 0; c < 3; ++c)
        rgb[3 * p + c] = value.w > 0 ? static_cast<unsigned char>(fminf(
                                           255.f, fmaxf(0.f, floorf(colors[c] / value.w + .5f))))
                                     : 0;
}
class CudaRenderer final : public IRenderer {
  public:
    std::string device_name() const override { return "cuda"; }
    Result render(const std::vector<Image>& images, const WarpPlan& plan,
                  const Options& o) const override {
        validate_plan(images, plan, o);
        check(cudaSetDevice(0), "CUDA device selection");
        const std::size_t count = static_cast<std::size_t>(plan.canvas.width) * plan.canvas.height;
        std::size_t largest = 0;
        for (const auto& image : images)
            largest = std::max(largest, image.pixels.size());
        DeviceBuffer<unsigned char> input(largest), output(count * 3), mask(count);
        DeviceBuffer<float> map_x(count), map_y(count);
        DeviceBuffer<float4> sums(count);
        check(cudaMemset(sums.get(), 0, count * sizeof(float4)), "CUDA accumulator initialization");
        constexpr int threads = 256;
        const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
        for (std::size_t i = 0; i < images.size(); ++i) {
            input.upload(images[i].pixels.data(), images[i].pixels.size());
            map_x.upload(plan.maps[i].x.data(), count);
            map_y.upload(plan.maps[i].y.data(), count);
            accumulate<<<blocks, threads>>>(input.get(), images[i].width, images[i].height,
                                            map_x.get(), map_y.get(), count,
                                            static_cast<float>(o.feather_width), sums.get());
            check(cudaGetLastError(), "CUDA accumulation launch");
            // All transfers and launches use this thread's default stream in sequence.
        }
        finish<<<blocks, threads>>>(sums.get(), count, output.get(), mask.get());
        check(cudaGetLastError(), "CUDA compositing launch");
        check(cudaDeviceSynchronize(), "CUDA renderer synchronization");
        Result result;
        result.canvas = plan.canvas;
        result.image = Image(plan.canvas.width, plan.canvas.height);
        result.mask.resize(count);
        output.download(result.image.pixels.data(), count * 3);
        mask.download(result.mask.data(), count);
        return result;
    }
};
} // namespace
std::unique_ptr<IRenderer> make_cuda_renderer() {
    return std::make_unique<CudaRenderer>();
}
} // namespace stitchkit::detail
namespace stitchkit {
bool cuda_available() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
} // namespace stitchkit
