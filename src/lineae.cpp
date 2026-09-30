#include "stitchkit/lineae.hpp"
#include "stitchkit/stitcher.hpp"
#include <algorithm>
#include <set>

namespace stitchkit {
LineaeNormalization lineae_normalization(const std::string& variant) {
    static const std::set<std::string> linea{"A", "F", "P", "N", "T"};
    static const std::set<std::string> imagenet{"S", "M", "L", "X", "XL", "2XL", "3XL"};
    if (linea.count(variant))
        return {{.538f, .494f, .453f}, {.257f, .263f, .273f}};
    if (imagenet.count(variant))
        return {{.485f, .456f, .406f}, {.229f, .224f, .225f}};
    throw Error("Unknown LINEAE variant; expected A/F/P/N/T/S/M/L/X/XL/2XL/3XL");
}
std::vector<float> prepare_lineae_input(const Image& image, Size size, const std::string& variant) {
    const auto profile = lineae_normalization(variant);
    const auto resized = resize_image(image, size);
    const auto count = resized.pixel_count();
    std::vector<float> tensor(3 * count);
    for (int c = 0; c < 3; ++c)
        for (std::size_t p = 0; p < count; ++p)
            tensor[c * count + p] = (resized.pixels[3 * p + c] / 255.f - profile.mean[c]) /
                                    profile.standard_deviation[c];
    return tensor;
}
namespace {
bool clip_segment(Point& a, Point& b, Size size) {
    const Point d = b - a;
    double start = 0, end = 1;
    const double p[4] = {-d.x, d.x, -d.y, d.y};
    const double q[4] = {a.x, size.width - 1 - a.x, a.y, size.height - 1 - a.y};
    for (int k = 0; k < 4; ++k) {
        if (std::abs(p[k]) < 1e-12) {
            if (q[k] < 0)
                return false;
            continue;
        }
        const double t = q[k] / p[k];
        if (p[k] < 0)
            start = std::max(start, t);
        else
            end = std::min(end, t);
        if (start > end)
            return false;
    }
    b = a + d * end;
    a = a + d * start;
    return true;
}
} // namespace
std::vector<Structure> decode_lineae_output(const std::vector<float>& logits,
                                            const std::vector<float>& lines, Size size,
                                            double threshold, int max_lines,
                                            double minimum_length) {
    if (logits.empty() || logits.size() % 2 || lines.size() != 2 * logits.size())
        throw Error("LINEAE output size mismatch");
    if (size.width < 2 || size.height < 2 || !std::isfinite(threshold) || threshold < 0 ||
        threshold > 1 || max_lines < 1 || !std::isfinite(minimum_length) || minimum_length <= 0)
        throw Error("Invalid LINEAE decoding options");
    std::vector<Structure> result;
    for (std::size_t k = 0; k < logits.size() / 2; ++k) {
        for (int j = 0; j < 2; ++j)
            if (!std::isfinite(logits[2 * k + j]))
                throw Error("Non-finite LINEAE logit");
        for (int j = 0; j < 4; ++j)
            if (!std::isfinite(lines[4 * k + j]))
                throw Error("Non-finite LINEAE endpoint");
        const double value = logits[2 * k];
        const double score =
            value >= 0 ? 1 / (1 + std::exp(-value)) : std::exp(value) / (1 + std::exp(value));
        if (score < threshold)
            continue;
        Point a{double(lines[4 * k]) * size.width, double(lines[4 * k + 1]) * size.height};
        Point b{double(lines[4 * k + 2]) * size.width, double(lines[4 * k + 3]) * size.height};
        if (!clip_segment(a, b, size))
            continue;
        const double length = std::sqrt((b - a).squared_norm());
        if (length < minimum_length)
            continue;
        Structure s;
        s.confidence = score;
        const int samples = std::clamp(static_cast<int>(std::ceil(length / 8)) + 1, 3, 24);
        for (int j = 0; j < samples; ++j)
            s.points.push_back(a + (b - a) * (double(j) / (samples - 1)));
        result.push_back(std::move(s));
    }
    std::stable_sort(result.begin(), result.end(), [](const Structure& a, const Structure& b) {
        return a.confidence > b.confidence;
    });
    if (result.size() > static_cast<std::size_t>(max_lines))
        result.resize(max_lines);
    return result;
}
} // namespace stitchkit
