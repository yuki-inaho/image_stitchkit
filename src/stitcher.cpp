#include "stitchkit/stitcher.hpp"
#include "internal.hpp"
#include <chrono>
#include <tuple>

namespace stitchkit {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
auto registration_key(const Options& o) {
    return std::make_tuple(o.max_features, o.ratio_threshold, o.ransac_threshold,
                           o.ransac_iterations, o.minimum_matches, o.seed);
}
void check_horizon(const Homography& h, Size size) {
    double sign = 0;
    for (auto p : std::array<Point, 4>{{{0, 0}, {double(size.width - 1), 0},
          {double(size.width - 1), double(size.height - 1)}, {0, double(size.height - 1)}}}) {
        const double denominator = h.values[6] * p.x + h.values[7] * p.y + h.values[8];
        if (!std::isfinite(denominator) || std::abs(denominator) < 1e-9 || !finite(h.apply(p)))
            throw Error("Homography intersects the projective horizon");
        if (sign * denominator < 0)
            throw Error("Homography crosses the image horizon");
        sign = denominator;
    }
}
} // namespace

struct PreparedScene::State {
    std::vector<Image> images;
    Registration registration;
    Options options;
    double registration_ms = 0;
    int feature_extractions = -1;
};
PreparedScene::PreparedScene(std::shared_ptr<const State> state) : state_(std::move(state)) {}
std::size_t PreparedScene::image_count() const { if (!state_) throw Error("Moved-from PreparedScene"); return state_->images.size(); }
double PreparedScene::registration_ms() const { if (!state_) throw Error("Moved-from PreparedScene"); return state_->registration_ms; }
int PreparedScene::feature_extractions() const { if (!state_) throw Error("Moved-from PreparedScene"); return state_->feature_extractions; }

std::unique_ptr<IWarpEstimator> make_warp_estimator(Method method,
                                                  std::shared_ptr<ILeastSquaresSolver> solver) {
    if (method == Method::rew)
        return detail::make_rew_estimator();
    if (method == Method::niswgsp || method == Method::gesgsp)
        return detail::make_mesh_estimator(method, std::move(solver));
    throw Error("Unknown warp method");
}
Stitcher::Stitcher(Options options)
    : options_(std::move(options)), matcher_(make_sift_matcher()), solver_(make_eigen_solver()) {
    options_.validate();
    if (options_.method == Method::gesgsp && options_.structures != StructureMode::none)
        detector_ = options_.structures == StructureMode::lineae ? make_lineae_detector(options_)
                                                                : make_edge_detector();
}
Stitcher::Stitcher(Options options, std::unique_ptr<IFeatureMatcher> matcher,
                   std::unique_ptr<IStructureDetector> detector,
                   std::shared_ptr<ILeastSquaresSolver> solver,
                   std::shared_ptr<IRenderer> renderer)
    : options_(std::move(options)), matcher_(std::move(matcher)), detector_(std::move(detector)),
      solver_(std::move(solver)), renderer_(std::move(renderer)) {
    options_.validate();
    if (!matcher_ || !solver_)
        throw Error("Stitcher requires a matcher and solver");
    if (options_.method == Method::gesgsp && options_.structures != StructureMode::none && !detector_)
        throw Error("GES-GSP requires a detector, or explicitly structures=none");
}
PreparedScene Stitcher::prepare(std::vector<Image> images) const {
    if (images.size() < 2 || images.size() > 16)
        throw Error("Expected 2 to 16 ordered, overlapping images");
    for (const auto& image : images)
        image.validate();
    auto state = std::make_shared<PreparedScene::State>();
    state->images = std::move(images);
    state->options = options_;
    auto& registration = state->registration;
    for (const auto& image : state->images)
        registration.sizes.push_back(image.size());
    registration.global_homographies.resize(state->images.size());
    registration.global_similarities.resize(state->images.size());
    const auto started = Clock::now();
    auto batch = matcher_->match_sequence(state->images, options_);
    if (batch.pairs.size() != state->images.size() - 1)
        throw Error("Matcher returned an invalid adjacent-pair count");
    state->feature_extractions = batch.feature_extractions;
    for (std::size_t i = 1; i < state->images.size(); ++i) {
        auto pair = std::move(batch.pairs[i - 1]);
        pair.source_index = i;
        pair.target_index = i - 1;
        if (pair.matches.size() < static_cast<std::size_t>(options_.minimum_matches))
            throw Error("Adjacent pair has insufficient verified correspondences");
        check_horizon(pair.homography, state->images[i].size());
        check_horizon(pair.similarity, state->images[i].size());
        registration.global_homographies[i] =
            stitchkit::compose(registration.global_homographies[i - 1], pair.homography);
        registration.global_similarities[i] =
            stitchkit::compose(registration.global_similarities[i - 1], pair.similarity);
        check_horizon(registration.global_homographies[i], state->images[i].size());
        registration.pairs.push_back(std::move(pair));
    }
    state->registration_ms = milliseconds(started);
    return PreparedScene(std::move(state));
}
Result Stitcher::compose_impl(const PreparedScene& scene, IRenderer& renderer,
                              std::vector<std::string> warnings, bool reused) const {
    if (!scene.state_)
        throw Error("Cannot compose a moved-from PreparedScene");
    if (registration_key(scene.state_->options) != registration_key(options_))
        throw Error("Registration options differ from the prepared scene; prepare it again");
    const auto& images = scene.state_->images;
    const auto& registration = scene.state_->registration;
    Diagnostics d;
    d.options = options_;
    d.input_sizes = registration.sizes;
    d.method = to_string(options_.method);
    d.requested_device = to_string(options_.device);
    d.actual_device = renderer.device_name();
    d.line_device = detector_ ? detector_->device_name() : "disabled";
    d.registration_reused = reused;
    d.prepared_registration_ms = scene.registration_ms();
    d.registration_ms = reused ? 0.0 : scene.registration_ms();
    d.feature_extractions = reused ? 0 : scene.feature_extractions();
    d.warnings = std::move(warnings);
    for (const auto& pair : registration.pairs)
        d.feature_matches += static_cast<int>(pair.matches.size());
    if (d.actual_device == "cuda")
        d.warnings.push_back("CUDA accelerates sampling/compositing only; features and geometry run on CPU.");
    auto started = Clock::now();
    std::vector<std::vector<Structure>> structures(images.size());
    if (options_.method == Method::gesgsp && detector_) {
        for (std::size_t i = 0; i < images.size(); ++i)
            structures[i] = detector_->detect(images[i], options_);
        d.line_device = detector_->device_name();
    }
    d.structure_ms = milliseconds(started);
    started = Clock::now();
    auto estimator = make_warp_estimator(options_.method, solver_);
    auto plan = estimator->estimate(registration, structures, options_, d);
    d.geometry_ms = milliseconds(started);
    started = Clock::now();
    auto result = renderer.render(images, plan, options_);
    d.render_ms = milliseconds(started);
    result.diagnostics = std::move(d);
    return result;
}
Result Stitcher::compose(const PreparedScene& scene) const {
    const auto started = Clock::now();
    std::vector<std::string> warnings;
    auto owned = renderer_ ? nullptr : make_renderer(options_.device, warnings);
    auto& renderer = renderer_ ? *renderer_ : *owned;
    auto result = compose_impl(scene, renderer, std::move(warnings), true);
    result.diagnostics.total_ms = milliseconds(started);
    return result;
}
Result Stitcher::stitch(std::vector<Image> images) const {
    const auto started = Clock::now();
    std::vector<std::string> warnings;
    // Select the backend before feature extraction, including explicit CUDA failure.
    auto owned = renderer_ ? nullptr : make_renderer(options_.device, warnings);
    auto& renderer = renderer_ ? *renderer_ : *owned;
    const auto scene = prepare(std::move(images));
    auto result = compose_impl(scene, renderer, std::move(warnings), false);
    result.diagnostics.total_ms = milliseconds(started);
    return result;
}
} // namespace stitchkit
