#include "internal.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <numeric>
#include <random>
#include <set>

namespace stitchkit::detail {
namespace {
Matrix3 normalization(const std::vector<Match>& matches, bool source) {
    Point mean;
    for (const auto& m : matches)
        mean = mean + (source ? m.source : m.target);
    mean = mean / double(matches.size());
    double distance = 0;
    for (const auto& m : matches)
        distance += std::sqrt(((source ? m.source : m.target) - mean).squared_norm());
    distance /= double(matches.size());
    if (!std::isfinite(distance) || distance < 1e-6)
        throw Error("Coincident correspondence points");
    const double s = std::sqrt(2.0) / distance;
    Matrix3 t = Matrix3::Identity();
    t(0, 0) = s;
    t(1, 1) = s;
    t(0, 2) = -s * mean.x;
    t(1, 2) = -s * mean.y;
    return t;
}
double residual(const Homography& h, const Match& match) {
    const auto p = h.apply(match.source);
    return finite(p) ? (p - match.target).squared_norm() : std::numeric_limits<double>::infinity();
}
struct RansacResult {
    Homography h;
    std::vector<int> inliers;
};
RansacResult ransac(const std::vector<Match>& matches, const Options& o, std::mt19937& rng,
                    int max_iterations) {
    RansacResult best;
    if (matches.size() < 4)
        return best;
    std::uniform_int_distribution<int> pick(0, static_cast<int>(matches.size()) - 1);
    double best_error = std::numeric_limits<double>::infinity();
    int iterations = max_iterations;
    for (int iter = 0; iter < iterations; ++iter) {
        std::array<int, 4> ids{};
        for (int j = 0; j < 4; ++j) {
            bool unique = false;
            while (!unique) {
                ids[j] = pick(rng);
                unique = true;
                for (int k = 0; k < j; ++k)
                    if (ids[k] == ids[j])
                        unique = false;
            }
        }
        std::vector<Match> sample;
        for (int index : ids)
            sample.push_back(matches[index]);
        bool degenerate = false;
        for (int a = 0; a < 4; ++a)
            for (int b = a + 1; b < 4; ++b)
                for (int c = b + 1; c < 4; ++c) {
                    if (std::abs(cross(sample[b].source - sample[a].source,
                                       sample[c].source - sample[a].source)) < 1.0 ||
                        std::abs(cross(sample[b].target - sample[a].target,
                                       sample[c].target - sample[a].target)) < 1.0)
                        degenerate = true;
                }
        if (degenerate)
            continue;
        Homography h;
        try {
            h = fit_homography(sample);
        } catch (const Error&) {
            continue;
        }
        std::vector<int> inliers;
        double error = 0;
        for (std::size_t i = 0; i < matches.size(); ++i) {
            const double e = residual(h, matches[i]);
            if (e <= o.ransac_threshold * o.ransac_threshold) {
                inliers.push_back(static_cast<int>(i));
                error += e;
            }
        }
        if (inliers.size() > best.inliers.size() ||
            (inliers.size() == best.inliers.size() && error < best_error)) {
            best = {h, std::move(inliers)};
            best_error = error;
            const double fraction = double(best.inliers.size()) / matches.size();
            const double miss = 1.0 - std::pow(fraction, 4);
            if (miss <= 1e-12)
                iterations = std::min(iterations, iter + 32);
            else if (miss < 1.0) {
                const double needed = std::log(0.001) / std::log(miss);
                if (std::isfinite(needed) && needed < iterations)
                    iterations = std::min(
                        iterations,
                        std::max(iter + 1, std::max(32, static_cast<int>(std::ceil(needed)))));
            }
        }
    }
    if (best.inliers.size() >= 4) {
        // Refit only the dominant consensus, never fit one homography to several depth planes.
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<Match> inliers;
            for (int i : best.inliers)
                inliers.push_back(matches[i]);
            try {
                best.h = fit_homography(inliers);
            } catch (const Error&) {
                break;
            }
            std::vector<int> refined;
            for (std::size_t i = 0; i < matches.size(); ++i)
                if (residual(best.h, matches[i]) <= o.ransac_threshold * o.ransac_threshold)
                    refined.push_back(static_cast<int>(i));
            if (refined.size() < 4)
                break;
            best.inliers = std::move(refined);
        }
    }
    return best;
}
} // namespace
Homography fit_homography(const std::vector<Match>& matches, const std::vector<double>& weights) {
    if (matches.size() < 4)
        throw Error("Homography needs at least four matches");
    if (!weights.empty() && weights.size() != matches.size())
        throw Error("Invalid DLT weights");
    for (const auto& m : matches)
        if (!finite(m.source) || !finite(m.target))
            throw Error("Non-finite match");
    const Matrix3 ts = normalization(matches, true), tt = normalization(matches, false);
    const Homography hs = homography(ts), ht = homography(tt);
    Eigen::Matrix<double, 9, 9> normal = Eigen::Matrix<double, 9, 9>::Zero();
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const Point a = hs.apply(matches[i].source), b = ht.apply(matches[i].target);
        Eigen::Matrix<double, 9, 1> x, y;
        x << -a.x, -a.y, -1, 0, 0, 0, b.x * a.x, b.x * a.y, b.x;
        y << 0, 0, 0, -a.x, -a.y, -1, b.y * a.x, b.y * a.y, b.y;
        const double w = weights.empty() ? 1.0 : weights[i];
        if (!std::isfinite(w) || w < 0)
            throw Error("Invalid DLT weight");
        normal.noalias() += w * w * (x * x.transpose() + y * y.transpose());
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eigen(normal);
    if (eigen.info() != Eigen::Success ||
        eigen.eigenvalues()(1) < 1e-10 * std::max(1.0, eigen.eigenvalues()(8)))
        throw Error("Degenerate homography correspondences");
    const auto v = eigen.eigenvectors().col(0);
    Matrix3 h;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            h(r, c) = v(3 * r + c);
    return homography(tt.inverse() * h * ts);
}
Homography fit_similarity(const std::vector<Match>& matches) {
    if (matches.size() < 2)
        throw Error("Similarity needs at least two matches");
    Eigen::MatrixXd a(matches.size() * 2, 4);
    Eigen::VectorXd b(matches.size() * 2);
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const auto& m = matches[i];
        a.row(2 * i) << m.source.x, -m.source.y, 1, 0;
        a.row(2 * i + 1) << m.source.y, m.source.x, 0, 1;
        b(2 * i) = m.target.x;
        b(2 * i + 1) = m.target.y;
    }
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a);
    if (qr.rank() != 4)
        throw Error("Degenerate similarity correspondences");
    const Eigen::Vector4d v = qr.solve(b);
    if (!v.allFinite() || std::hypot(v(0), v(1)) < 0.02 || std::hypot(v(0), v(1)) > 50)
        throw Error("Implausible similarity scale");
    Matrix3 h;
    h << v(0), -v(1), v(2), v(1), v(0), v(3), 0, 0, 1;
    return homography(h);
}
PairRegistration robust_register(std::vector<Match> matches, const Options& o) {
    o.validate();
    PairRegistration out;
    out.tentative_matches = static_cast<int>(matches.size());
    if (matches.size() < static_cast<std::size_t>(o.minimum_matches))
        throw Error("Insufficient feature matches; images may not overlap");
    std::mt19937 rng(o.seed);
    const auto primary = ransac(matches, o, rng, o.ransac_iterations);
    if (primary.inliers.size() < static_cast<std::size_t>(o.minimum_matches))
        throw Error("No reliable geometric consensus; images may not overlap");
    out.homography = primary.h;
    out.dominant_inliers = static_cast<int>(primary.inliers.size());
    std::vector<Match> dominant;
    std::vector<bool> selected(matches.size(), false);
    for (int i : primary.inliers) {
        selected[i] = true;
        dominant.push_back(matches[i]);
    }
    out.similarity = fit_similarity(dominant);
    double sum = 0;
    for (const auto& m : dominant)
        sum += residual(out.homography, m);
    out.reprojection_rmse = std::sqrt(sum / dominant.size());
    // Sequential consensus retains secondary depth planes instead of discarding all parallax.
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<Match> rest;
        std::vector<int> original;
        for (std::size_t i = 0; i < matches.size(); ++i)
            if (!selected[i]) {
                rest.push_back(matches[i]);
                original.push_back(static_cast<int>(i));
            }
        if (rest.size() < static_cast<std::size_t>(o.minimum_matches))
            break;
        auto next = ransac(rest, o, rng, std::min(o.ransac_iterations, 600));
        if (next.inliers.size() < static_cast<std::size_t>(o.minimum_matches))
            break;
        for (int i : next.inliers)
            selected[original[i]] = true;
    }
    for (std::size_t i = 0; i < matches.size(); ++i)
        if (selected[i])
            out.matches.push_back(matches[i]);
    return out;
}
std::vector<Match> apap_matches(const PairRegistration& pair, const Mesh& source, Size target,
                                const Options& o) {
    std::vector<Match> out;
    if (!o.use_apap || pair.matches.size() < 6)
        return out;
    double x0 = 1e30, y0 = 1e30, x1 = -1e30, y1 = -1e30;
    for (const auto& m : pair.matches) {
        x0 = std::min(x0, m.source.x);
        x1 = std::max(x1, m.source.x);
        y0 = std::min(y0, m.source.y);
        y1 = std::max(y1, m.source.y);
    }
    for (Point p : source.original) {
        if (p.x < x0 || p.x > x1 || p.y < y0 || p.y > y1)
            continue;
        std::vector<double> weights;
        weights.reserve(pair.matches.size());
        for (const auto& m : pair.matches)
            weights.push_back(std::max(
                0.0015, std::exp(-(p - m.source).squared_norm() / (o.apap_sigma * o.apap_sigma))));
        try {
            const auto h = fit_homography(pair.matches, weights);
            const Point q = h.apply(p);
            if (finite(q) && q.x >= 0 && q.y >= 0 && q.x <= target.width - 1 &&
                q.y <= target.height - 1)
                out.push_back({p, q, 0.25});
        } catch (const Error&) { /* Degenerate local support contributes no extra alignment row. */
        }
    }
    return out;
}
} // namespace stitchkit::detail
