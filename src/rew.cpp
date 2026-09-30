#include "internal.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <numeric>
#include <set>

namespace stitchkit::detail {
namespace {
double radial(double squared_radius) {
    return squared_radius > 1e-20 ? .5 * squared_radius * std::log(squared_radius) : 0.;
}
struct ElasticField {
    Homography inverse_h;
    std::vector<Point> controls;
    Eigen::MatrixXd coefficients;
    Point center;
    double scale = 1;
    Point overlap_min, overlap_max;
    double fade_width = 1, max_residual = 0;
    Point evaluate(Point target) const {
        const Point projected = inverse_h.apply(target);
        if (!finite(projected))
            return projected;
        const Point q = (projected - center) / scale;
        const int n = static_cast<int>(controls.size());
        Point residual{
            coefficients(n, 0) + coefficients(n + 1, 0) * q.x + coefficients(n + 2, 0) * q.y,
            coefficients(n, 1) + coefficients(n + 1, 1) * q.x + coefficients(n + 2, 1) * q.y};
        for (int k = 0; k < n; ++k) {
            const double basis = radial((q - controls[k]).squared_norm());
            residual.x += basis * coefficients(k, 0);
            residual.y += basis * coefficients(k, 1);
        }
        const double outside =
            std::max({0., overlap_min.x - projected.x, projected.x - overlap_max.x,
                      overlap_min.y - projected.y, projected.y - overlap_max.y});
        const double eta = std::clamp(1 - outside / fade_width, 0., 1.);
        return projected - residual * eta;
    }
};
ElasticField fit_field(const PairRegistration& pair, Size source_size, Size target_size,
                       const Options& o, Diagnostics& diagnostic) {
    ElasticField field;
    field.inverse_h = pair.homography.inverse();
    field.center = {.5 * (source_size.width - 1), .5 * (source_size.height - 1)};
    field.scale =
        std::hypot(static_cast<double>(source_size.width), static_cast<double>(source_size.height));
    std::vector<Point> raw, residuals;
    std::set<std::pair<int, int>> seen;
    for (const auto& m : pair.matches) {
        const auto q = field.inverse_h.apply(m.target);
        if (!finite(q))
            continue;
        if (!seen.emplace(static_cast<int>(std::lround(q.x)), static_cast<int>(std::lround(q.y)))
                 .second)
            continue;
        raw.push_back(q);
        residuals.push_back(q - m.source);
    }
    if (raw.size() < 4)
        throw Error("Insufficient distinct REW controls");
    if (raw.size() > static_cast<std::size_t>(o.max_tps_controls)) {
        // Deterministic farthest-point coverage, rather than keeping one textured corner.
        std::vector<Point> selected, selected_residual;
        std::vector<double> distances(raw.size(), std::numeric_limits<double>::infinity());
        std::vector<bool> used(raw.size(), false);
        std::size_t next = 0;
        for (int k = 0; k < o.max_tps_controls; ++k) {
            selected.push_back(raw[next]);
            selected_residual.push_back(residuals[next]);
            used[next] = true;
            double best = -1;
            std::size_t candidate = 0;
            for (std::size_t j = 0; j < raw.size(); ++j) {
                distances[j] = std::min(distances[j], (raw[j] - raw[next]).squared_norm());
                if (!used[j] && distances[j] > best) {
                    best = distances[j];
                    candidate = j;
                }
            }
            next = candidate;
        }
        raw = std::move(selected);
        residuals = std::move(selected_residual);
    }
    const double lambda = o.tps_lambda * target_size.width * target_size.height /
                          (field.scale * field.scale) * 8 * 3.14159265358979323846;
    for (int iteration = 0; iteration < 10; ++iteration) {
        const int n = static_cast<int>(raw.size());
        field.controls.clear();
        for (auto q : raw)
            field.controls.push_back((q - field.center) / field.scale);
        Eigen::MatrixXd a = Eigen::MatrixXd::Zero(n + 3, n + 3),
                        b = Eigen::MatrixXd::Zero(n + 3, 2);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                a(i, j) = radial((field.controls[i] - field.controls[j]).squared_norm());
            a(i, i) += lambda;
            a(i, n) = a(n, i) = 1;
            a(i, n + 1) = a(n + 1, i) = field.controls[i].x;
            a(i, n + 2) = a(n + 2, i) = field.controls[i].y;
            b(i, 0) = residuals[i].x;
            b(i, 1) = residuals[i].y;
        }
        Eigen::FullPivLU<Eigen::MatrixXd> lu(a);
        if (lu.rank() != n + 3)
            throw Error("Degenerate thin-plate-spline control configuration");
        field.coefficients = lu.solve(b);
        if (!field.coefficients.allFinite() ||
            (a * field.coefficients - b).norm() > 1e-7 * std::max(1., b.norm()))
            throw Error("Thin-plate-spline solve failed");
        if (iteration == 9 || n < 12)
            break;
        const auto weights = field.coefficients.topRows(n);
        const double sx = std::sqrt(weights.col(0).array().square().mean());
        const double sy = std::sqrt(weights.col(1).array().square().mean());
        std::vector<Point> keep, keep_residual;
        for (int i = 0; i < n; ++i) {
            const bool outlier = (sx > 1e-9 && std::abs(weights(i, 0)) > 3 * sx) ||
                                 (sy > 1e-9 && std::abs(weights(i, 1)) > 3 * sy);
            if (!outlier) {
                keep.push_back(raw[i]);
                keep_residual.push_back(residuals[i]);
            }
        }
        if (keep.size() == raw.size() || keep.size() < 6)
            break;
        raw = std::move(keep);
        residuals = std::move(keep_residual);
    }
    Point low{1e30, 1e30}, high{-1e30, -1e30};
    for (const Point p :
         std::array<Point, 4>{{{0, 0},
                               {double(target_size.width - 1), 0},
                               {double(target_size.width - 1), double(target_size.height - 1)},
                               {0, double(target_size.height - 1)}}}) {
        const auto q = field.inverse_h.apply(p);
        if (!finite(q))
            throw Error("Invalid REW overlap boundary");
        low = {std::min(low.x, q.x), std::min(low.y, q.y)};
        high = {std::max(high.x, q.x), std::max(high.y, q.y)};
    }
    field.overlap_min = {std::max(0., low.x), std::max(0., low.y)};
    field.overlap_max = {std::min(double(source_size.width - 1), high.x),
                         std::min(double(source_size.height - 1), high.y)};
    if (field.overlap_min.x >= field.overlap_max.x || field.overlap_min.y >= field.overlap_max.y)
        throw Error("No geometric overlap for REW");
    for (const auto d : residuals)
        field.max_residual = std::max({field.max_residual, std::abs(d.x), std::abs(d.y)});
    field.overlap_min = field.overlap_min - Point{field.max_residual, field.max_residual};
    field.overlap_max = field.overlap_max + Point{field.max_residual, field.max_residual};
    field.fade_width =
        std::max(1., 5 * field.max_residual); // Exact zero-parallax case is well-defined.
    diagnostic.tps_controls += static_cast<int>(field.controls.size());
    diagnostic.equations += 2 * (static_cast<int>(field.controls.size()) + 3);
    diagnostic.unknowns += 2 * (static_cast<int>(field.controls.size()) + 3);
    return field;
}
class RewEstimator final : public IWarpEstimator {
  public:
    WarpPlan estimate(const Registration& r, const std::vector<std::vector<Structure>>&,
                      const Options& o, Diagnostics& d) const override {
        if (r.sizes.size() < 2 || r.pairs.size() + 1 != r.sizes.size() ||
            r.global_homographies.size() != r.sizes.size())
            throw Error("REW requires an ordered, connected image chain");
        std::vector<ElasticField> fields;
        double before = 0, after = 0;
        int count = 0;
        double padding = 0;
        for (std::size_t k = 0; k < r.pairs.size(); ++k) {
            const auto& pair = r.pairs[k];
            if (pair.target_index != k || pair.source_index != k + 1)
                throw Error("REW pair ordering is inconsistent");
            fields.push_back(fit_field(pair, r.sizes[k + 1], r.sizes[k], o, d));
            for (const auto& m : pair.matches) {
                before += (fields.back().inverse_h.apply(m.target) - m.source).squared_norm();
                after += (fields.back().evaluate(m.target) - m.source).squared_norm();
                ++count;
            }
            const double global_scale = std::hypot(r.global_homographies[k + 1].values[0],
                                                   r.global_homographies[k + 1].values[3]);
            padding += 2 * fields.back().max_residual * std::clamp(global_scale, .1, 10.);
        }
        d.alignment_rmse_before = count ? std::sqrt(before / count) : 0;
        d.alignment_rmse_after = count ? std::sqrt(after / count) : 0;
        std::vector<Point> corners;
        for (std::size_t i = 0; i < r.sizes.size(); ++i) {
            const double w = r.sizes[i].width - 1, h = r.sizes[i].height - 1;
            for (auto p : std::array<Point, 4>{{{0, 0}, {w, 0}, {w, h}, {0, h}}})
                corners.push_back(r.global_homographies[i].apply(p));
        }
        WarpPlan plan;
        plan.canvas = bounded_canvas(corners, o.max_canvas_pixels, std::ceil(padding));
        const auto pixels = static_cast<std::size_t>(plan.canvas.width) * plan.canvas.height;
        if (pixels * r.sizes.size() > 64'000'000)
            throw Error("REW maps exceed the 64M-sample working-memory limit");
        constexpr int step = 8;
        const int gx = (plan.canvas.width - 1 + step - 1) / step,
                  gy = (plan.canvas.height - 1 + step - 1) / step;
        const double sx = double(plan.canvas.width - 1) / gx,
                     sy = double(plan.canvas.height - 1) / gy;
        for (std::size_t i = 0; i < r.sizes.size(); ++i) {
            std::vector<Point> coarse(static_cast<std::size_t>(gx + 1) * (gy + 1));
            for (int y = 0; y <= gy; ++y)
                for (int x = 0; x <= gx; ++x) {
                    Point q = plan.canvas.origin + Point{x * sx, y * sy};
                    for (std::size_t k = 0; k < i; ++k)
                        q = fields[k].evaluate(q);
                    coarse[static_cast<std::size_t>(y) * (gx + 1) + x] = q;
                }
            SamplingMap map;
            map.x.assign(pixels, -1);
            map.y.assign(pixels, -1);
            for (int y = 0; y < plan.canvas.height; ++y)
                for (int x = 0; x < plan.canvas.width; ++x) {
                    const int cx = std::min(gx - 1, int(x / sx)),
                              cy = std::min(gy - 1, int(y / sy));
                    const double u = x / sx - cx, v = y / sy - cy;
                    const auto idx = static_cast<std::size_t>(cy) * (gx + 1) + cx;
                    Point q = coarse[idx] * ((1 - u) * (1 - v)) + coarse[idx + 1] * (u * (1 - v)) +
                              coarse[idx + gx + 2] * (u * v) + coarse[idx + gx + 1] * ((1 - u) * v);
                    if (finite(q) && q.x >= -1e-4 && q.y >= -1e-4 &&
                        q.x <= r.sizes[i].width - 1 + 1e-4 && q.y <= r.sizes[i].height - 1 + 1e-4) {
                        const auto p = static_cast<std::size_t>(y) * plan.canvas.width + x;
                        map.x[p] =
                            static_cast<float>(std::clamp(q.x, 0., double(r.sizes[i].width - 1)));
                        map.y[p] =
                            static_cast<float>(std::clamp(q.y, 0., double(r.sizes[i].height - 1)));
                    }
                }
            plan.maps.push_back(std::move(map));
        }
        if (r.sizes.size() > 2)
            d.warnings.push_back("REW multi-image mode composes pairwise inverse fields; no global "
                                 "multi-view refinement is performed.");
        return plan;
    }
};
} // namespace
std::unique_ptr<IWarpEstimator> make_rew_estimator() {
    return std::make_unique<RewEstimator>();
}
} // namespace stitchkit::detail
