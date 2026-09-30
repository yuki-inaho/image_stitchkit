#include "stitchkit/stitcher.hpp"
#include <algorithm>
#include <iostream>
#include <set>

namespace {
void usage() {
    std::cout << R"(stitchkit 0.1.0 -- ordered, overlapping PNG/JPEG images
Usage: stitch [options] --output panorama.png image0.png image1.png [image2.png ...]
  --method rew|niswgsp|gesgsp       Warp method (default gesgsp)
  --device cpu|cuda|auto           Rendering backend; explicit cuda never silently falls back
  --report report.json            Machine-readable diagnostics
  --max-side N                    Resize each input to at most N pixels per side (default 1000; 0 disables)
  --grid N                        Mesh cell size (default 32)
  --max-features N                 SIFT feature limit per image (default 2500)
  --ratio X                       Mutual descriptor ratio threshold (default 0.78)
  --ransac-threshold X             Primary/secondary model inlier threshold in pixels
  --no-apap                       Disable mesh-vertex moving-DLT correspondences
  --structures edges|lineae|none   GES-GSP structure detector (default edges)
  --structure-weight X            Residual multiplier for geometric structure equations
  --tps-lambda X                  REW smoothing strength (default 0.001)
  --seed N                        Deterministic registration seed
  --lineae-model path.onnx         Model file (not downloaded automatically)
  --lineae-variant A|F|P|N|T|S|M|L|X|XL|2XL|3XL
  --lineae-device cpu|cuda|auto    ONNX Runtime execution provider, independent of renderer
  --lineae-threshold X             Sigmoid class-0 confidence threshold (default 0.3)
  --capabilities                  Print compiled/runtime capabilities as JSON
  --version                       Print version
  --help                          Print this help
All geometry and SIFT run on CPU. CUDA optionally accelerates sampling/compositing.
Input image order must follow an overlapping chain; no automatic unordered graph discovery.
)";
}
int integer(const std::string& text, const std::string& name) {
    std::size_t used = 0;
    int value = 0;
    try {
        value = std::stoi(text, &used);
    } catch (const std::exception&) {
        throw stitchkit::Error("Invalid integer for " + name);
    }
    if (used != text.size())
        throw stitchkit::Error("Invalid integer for " + name);
    return value;
}
double real(const std::string& text, const std::string& name) {
    std::size_t used = 0;
    double value = 0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        throw stitchkit::Error("Invalid number for " + name);
    }
    if (used != text.size() || !std::isfinite(value))
        throw stitchkit::Error("Invalid number for " + name);
    return value;
}
} // namespace
int main(int argc, char** argv) {
    try {
        stitchkit::Options options;
        std::filesystem::path output, report;
        std::vector<std::filesystem::path> paths;
        int max_side = 1000;
        std::set<std::string> seen;
        bool positional = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (!positional && arg == "--") {
                positional = true;
                continue;
            }
            if (positional || arg.empty() || arg[0] != '-') {
                paths.emplace_back(arg);
                continue;
            }
            if (!seen.insert(arg).second)
                throw stitchkit::Error("Duplicate option: " + arg);
            if (arg == "--help") {
                usage();
                return 0;
            }
            if (arg == "--version") {
                std::cout << stitchkit::version() << '\n';
                return 0;
            }
            if (arg == "--capabilities") {
                std::cout << "{\"cuda_compiled\":"
                          << (stitchkit::cuda_compiled() ? "true" : "false")
                          << ",\"cuda_available\":"
                          << (stitchkit::cuda_available() ? "true" : "false")
                          << ",\"lineae_compiled\":"
                          << (stitchkit::lineae_compiled() ? "true" : "false") << "}\n";
                return 0;
            }
            if (arg == "--no-apap") {
                options.use_apap = false;
                continue;
            }
            if (i + 1 >= argc)
                throw stitchkit::Error("Missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--method")
                options.method = stitchkit::parse_method(value);
            else if (arg == "--device")
                options.device = stitchkit::parse_device(value);
            else if (arg == "--output")
                output = value;
            else if (arg == "--report")
                report = value;
            else if (arg == "--max-side")
                max_side = integer(value, arg);
            else if (arg == "--grid")
                options.grid_size = integer(value, arg);
            else if (arg == "--max-features")
                options.max_features = integer(value, arg);
            else if (arg == "--ratio")
                options.ratio_threshold = real(value, arg);
            else if (arg == "--ransac-threshold")
                options.ransac_threshold = real(value, arg);
            else if (arg == "--structure-weight")
                options.structure_weight = real(value, arg);
            else if (arg == "--tps-lambda")
                options.tps_lambda = real(value, arg);
            else if (arg == "--seed") {
                const int seed = integer(value, arg);
                if (seed < 0)
                    throw stitchkit::Error("Seed must be nonnegative");
                options.seed = static_cast<std::uint32_t>(seed);
            } else if (arg == "--structures") {
                if (value == "edges")
                    options.structures = stitchkit::StructureMode::edges;
                else if (value == "lineae")
                    options.structures = stitchkit::StructureMode::lineae;
                else if (value == "none")
                    options.structures = stitchkit::StructureMode::none;
                else
                    throw stitchkit::Error("Unknown structure detector: " + value);
            } else if (arg == "--lineae-model")
                options.lineae_model = value;
            else if (arg == "--lineae-variant")
                options.lineae_variant = value;
            else if (arg == "--lineae-device")
                options.lineae_device = stitchkit::parse_device(value);
            else if (arg == "--lineae-threshold")
                options.lineae_threshold = real(value, arg);
            else
                throw stitchkit::Error("Unknown option: " + arg);
        }
        options.validate();
        if (output.empty() || paths.size() < 2)
            throw stitchkit::Error("Provide --output and at least two input paths; see --help");
        if (max_side != 0 && (max_side < 64 || max_side > 32768))
            throw stitchkit::Error("--max-side must be 0 or in [64,32768]");
        std::string ext = output.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return char(std::tolower(c)); });
        if (ext != ".png")
            throw stitchkit::Error("Output must have a .png extension");
        const auto normal = [](const std::filesystem::path& p) {
            return std::filesystem::weakly_canonical(std::filesystem::absolute(p));
        };
        for (const auto& path : paths)
            if (normal(path) == normal(output) ||
                (!report.empty() && normal(path) == normal(report)))
                throw stitchkit::Error("Output/report paths must not overwrite input images");
        if (!report.empty() && normal(output) == normal(report))
            throw stitchkit::Error("Output and report paths must differ");
        // Construction validates model configuration before decoding input assets.
        const stitchkit::Stitcher stitcher(options);
        std::vector<stitchkit::Image> images;
        for (const auto& path : paths) {
            auto image = stitchkit::read_image(path);
            images.push_back(max_side ? stitchkit::limit_size(image, max_side) : std::move(image));
        }
        auto result = stitcher.stitch(images);
        stitchkit::write_png(output, result.image, result.mask);
        if (!report.empty())
            stitchkit::write_report(report, result);
        std::cout << "method=" << result.diagnostics.method
                  << " renderer=" << result.diagnostics.actual_device
                  << " canvas=" << result.image.width << 'x' << result.image.height
                  << " matches=" << result.diagnostics.feature_matches
                  << " alignment_rmse=" << result.diagnostics.alignment_rmse_after << " px\n";
        for (const auto& warning : result.diagnostics.warnings)
            std::cerr << "warning: " << warning << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "stitch: " << e.what() << '\n';
        return 2;
    }
}
