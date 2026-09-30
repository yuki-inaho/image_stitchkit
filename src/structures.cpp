#include "internal.hpp"
#include <algorithm>
#include <numeric>

namespace stitchkit {
namespace {
class EdgeDetector final : public IStructureDetector {
  public:
    std::vector<Structure> detect(const Image& image, const Options& o) const override {
        image.validate();
        const int w = image.width, h = image.height;
        const auto gray = detail::grayscale(image);
        std::vector<float> magnitude(image.pixel_count(), 0), dx(image.pixel_count(), 0),
            dy(image.pixel_count(), 0);
        for (int y = 1; y < h - 1; ++y)
            for (int x = 1; x < w - 1; ++x) {
                const auto p = static_cast<std::size_t>(y) * w + x;
                dx[p] = -gray[p - w - 1] + gray[p - w + 1] - 2 * gray[p - 1] + 2 * gray[p + 1] -
                        gray[p + w - 1] + gray[p + w + 1];
                dy[p] = -gray[p - w - 1] - 2 * gray[p - w] - gray[p - w + 1] + gray[p + w - 1] +
                        2 * gray[p + w] + gray[p + w + 1];
                magnitude[p] = std::hypot(dx[p], dy[p]);
            }
        std::vector<std::uint8_t> edges(image.pixel_count(), 0), visited(image.pixel_count(), 0);
        for (int y = 2; y < h - 2; ++y)
            for (int x = 2; x < w - 2; ++x) {
                const auto p = static_cast<std::size_t>(y) * w + x;
                const float mag = magnitude[p];
                if (mag < 55)
                    continue;
                const float ax = std::abs(dx[p]), ay = std::abs(dy[p]);
                int offset = 0;
                if (ay < .41421356f * ax)
                    offset = 1;
                else if (ax < .41421356f * ay)
                    offset = w;
                else
                    offset = (dx[p] * dy[p] > 0) ? w + 1 : w - 1;
                if (mag >= magnitude[p - offset] && mag >= magnitude[p + offset])
                    edges[p] = 1;
            }
        const int offsets[8] = {-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1};
        auto next_neighbours = [&](int index) {
            std::vector<int> out;
            for (int delta : offsets)
                if (edges[index + delta] && !visited[index + delta])
                    out.push_back(index + delta);
            return out;
        };
        std::vector<int> starts;
        for (int y = 2; y < h - 2; ++y)
            for (int x = 2; x < w - 2; ++x) {
                const int p = y * w + x;
                if (!edges[p])
                    continue;
                int degree = 0;
                for (int k : offsets)
                    degree += edges[p + k];
                if (degree <= 1)
                    starts.push_back(p);
            }
        // Endpoint-first traces retain open edges; a second pass also retains closed curves.
        for (int y = 2; y < h - 2; ++y)
            for (int x = 2; x < w - 2; ++x)
                if (edges[y * w + x])
                    starts.push_back(y * w + x);
        struct Candidate {
            Structure structure;
            double score;
        };
        std::vector<Candidate> candidates;
        for (int start : starts) {
            if (visited[start])
                continue;
            std::vector<Point> chain;
            int current = start, previous = -1;
            double total_strength = 0;
            while (!visited[current]) {
                visited[current] = 1;
                chain.push_back({double(current % w), double(current / w)});
                total_strength += magnitude[current];
                const auto ns = next_neighbours(current);
                if (ns.empty())
                    break;
                int next = ns.front();
                double best = -1e30;
                for (int q : ns) {
                    double score = magnitude[q] * .001;
                    if (previous >= 0) {
                        const Point a{double(current % w - previous % w),
                                      double(current / w - previous / w)};
                        const Point b{double(q % w - current % w), double(q / w - current / w)};
                        score += dot(a, b) / std::sqrt(a.squared_norm() * b.squared_norm());
                    }
                    if (score > best) {
                        best = score;
                        next = q;
                    }
                }
                previous = current;
                current = next;
            }
            if (chain.size() < 24)
                continue;
            double length = 0;
            for (std::size_t k = 1; k < chain.size(); ++k)
                length += std::sqrt((chain[k] - chain[k - 1]).squared_norm());
            if (length < 32)
                continue;
            Structure s;
            s.confidence = std::clamp(total_strength / (chain.size() * 400.), .1, 1.);
            // Long contours are split to avoid constraining unrelated corners as one rigid shape.
            // Within each segment, curved samples are retained rather than replaced by a line.
            constexpr std::size_t max_chain = 180;
            for (std::size_t begin = 0; begin + 16 < chain.size(); begin += max_chain - 1) {
                const std::size_t end = std::min(chain.size() - 1, begin + max_chain - 1);
                s.points.clear();
                const int samples = std::clamp(static_cast<int>((end - begin) / 8) + 1, 3, 24);
                for (int k = 0; k < samples; ++k)
                    s.points.push_back(chain[begin + (end - begin) * k / (samples - 1)]);
                candidates.push_back({s, (end - begin) * s.confidence});
                if (end == chain.size() - 1)
                    break;
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
        std::vector<Structure> result;
        for (const auto& item : candidates) {
            if (result.size() >= static_cast<std::size_t>(o.max_structures))
                break;
            result.push_back(item.structure);
        }
        return result;
    }
};
} // namespace
std::unique_ptr<IStructureDetector> make_edge_detector() {
    return std::make_unique<EdgeDetector>();
}
bool lineae_compiled() noexcept {
#ifdef STITCHKIT_WITH_LINEAE
    return true;
#else
    return false;
#endif
}
std::unique_ptr<IStructureDetector> make_lineae_detector(const Options& o) {
#ifdef STITCHKIT_WITH_LINEAE
    return detail::make_onnx_lineae(o);
#else
    (void)o;
    throw Error("LINEAE requested but this build has no ONNX Runtime support");
#endif
}
} // namespace stitchkit
