#include "internal.hpp"
#include "vl/sift.h"
#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>

namespace stitchkit {
namespace {
struct Feature {
    Point point;
    double scale;
    std::array<float, 128> descriptor;
};
std::vector<Feature> features(const Image& image, int limit) {
    if (image.width < 16 || image.height < 16)
        throw Error("SIFT requires both image dimensions to be at least 16 pixels");
    // VLFeat 0.9.20 uses a lazy process-wide exp lookup table. Protect initialization
    // and execution rather than expose a data race to callers of different Stitcher instances.
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    const auto gray = detail::grayscale(image);
    using Sift = std::unique_ptr<VlSiftFilt, decltype(&vl_sift_delete)>;
    Sift sift(vl_sift_new(image.width, image.height, -1, 3, 0), &vl_sift_delete);
    if (!sift)
        throw Error("Cannot allocate SIFT detector");
    vl_sift_set_peak_thresh(sift.get(), 2.0);
    vl_sift_set_edge_thresh(sift.get(), 10.0);
    std::vector<Feature> found;
    int status = vl_sift_process_first_octave(sift.get(), gray.data());
    while (status == VL_ERR_OK) {
        vl_sift_detect(sift.get());
        const auto* points = vl_sift_get_keypoints(sift.get());
        const int count = vl_sift_get_nkeypoints(sift.get());
        for (int i = 0; i < count; ++i) {
            if (points[i].x < 1 || points[i].y < 1 || points[i].x > image.width - 2 ||
                points[i].y > image.height - 2)
                continue;
            double angles[4];
            const int orientations =
                vl_sift_calc_keypoint_orientations(sift.get(), angles, &points[i]);
            for (int j = 0; j < orientations; ++j) {
                Feature f;
                f.point = {points[i].x, points[i].y};
                f.scale = points[i].sigma;
                vl_sift_calc_keypoint_descriptor(sift.get(), f.descriptor.data(), &points[i],
                                                 angles[j]);
                float sum = 0;
                for (float v : f.descriptor)
                    sum += std::abs(v);
                if (sum < 1e-8f)
                    continue;
                for (float& v : f.descriptor)
                    v = std::sqrt(std::max(0.0f, v) / sum); // RootSIFT.
                found.push_back(f);
            }
        }
        status = vl_sift_process_next_octave(sift.get());
    }
    if (status != VL_ERR_EOF)
        throw Error("SIFT octave processing failed");
    if (found.size() > static_cast<std::size_t>(limit)) {
        // Spatially balanced selection prevents one textured corner from taking all slots.
        std::array<std::vector<Feature>, 64> buckets;
        std::stable_sort(found.begin(), found.end(),
                         [](const Feature& a, const Feature& b) { return a.scale > b.scale; });
        for (auto& f : found) {
            const int x = std::min(7, int(8 * f.point.x / image.width)),
                      y = std::min(7, int(8 * f.point.y / image.height));
            buckets[y * 8 + x].push_back(std::move(f));
        }
        found.clear();
        found.reserve(limit);
        for (std::size_t round = 0; found.size() < static_cast<std::size_t>(limit); ++round) {
            bool any = false;
            for (auto& bucket : buckets)
                if (round < bucket.size() && found.size() < static_cast<std::size_t>(limit)) {
                    found.push_back(std::move(bucket[round]));
                    any = true;
                }
            if (!any)
                break;
        }
    }
    return found;
}
struct Nearest {
    float best = 1e30f, second = 1e30f;
    int index = -1;
    void consider(float d, int i) {
        if (d < best) {
            second = best;
            best = d;
            index = i;
        } else if (d < second)
            second = d;
    }
};
PairRegistration match_features(const std::vector<Feature>& a,
                                const std::vector<Feature>& b, const Options& options) {
        if (a.size() < 2 || b.size() < 2)
            throw Error("Insufficient SIFT features; input may be blank or too small");
        std::vector<Nearest> ab(a.size()), ba(b.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; j < b.size(); ++j) {
                float d = 0;
                for (int k = 0; k < 128; ++k) {
                    const float delta = a[i].descriptor[k] - b[j].descriptor[k];
                    d += delta * delta;
                }
                ab[i].consider(d, static_cast<int>(j));
                ba[j].consider(d, static_cast<int>(i));
            }
        const double ratio2 = options.ratio_threshold * options.ratio_threshold;
        std::vector<Match> matches;
        std::set<std::tuple<int, int, int, int>> unique;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const auto& n = ab[i];
            if (n.index < 0)
                continue;
            const auto& r = ba[n.index];
            if (n.best >= ratio2 * n.second || r.index != static_cast<int>(i) ||
                r.best >= ratio2 * r.second)
                continue;
            const Point p = a[i].point, q = b[n.index].point;
            const auto key = std::make_tuple(int(std::lround(p.x)), int(std::lround(p.y)),
                                             int(std::lround(q.x)), int(std::lround(q.y)));
            if (unique.insert(key).second)
                matches.push_back({p, q, 1.0});
        }
        return detail::robust_register(std::move(matches), options);
}
class SiftMatcher final : public IFeatureMatcher {
  public:
    PairRegistration match(const Image& source, const Image& target,
                           const Options& options) const override {
        options.validate();
        source.validate();
        target.validate();
        return match_features(features(source, options.max_features),
                              features(target, options.max_features), options);
    }
    MatchBatch match_sequence(const std::vector<Image>& images,
                              const Options& options) const override {
        options.validate();
        if (images.size() < 2 || images.size() > 16)
            throw Error("Expected 2 to 16 images");
        for (const auto& image : images)
            image.validate();
        MatchBatch batch;
        batch.feature_extractions = static_cast<int>(images.size());
        auto previous = features(images.front(), options.max_features);
        for (std::size_t i = 1; i < images.size(); ++i) {
            auto current = features(images[i], options.max_features);
            batch.pairs.push_back(match_features(current, previous, options));
            previous = std::move(current); // Keep at most two descriptor sets alive.
        }
        return batch;
    }
};
} // namespace
MatchBatch IFeatureMatcher::match_sequence(const std::vector<Image>& images,
                                           const Options& options) const {
    if (images.size() < 2 || images.size() > 16)
        throw Error("Expected 2 to 16 images");
    MatchBatch batch;
    for (std::size_t i = 1; i < images.size(); ++i)
        batch.pairs.push_back(match(images[i], images[i - 1], options));
    return batch;
}
std::unique_ptr<IFeatureMatcher> make_sift_matcher() {
    return std::make_unique<SiftMatcher>();
}
} // namespace stitchkit
