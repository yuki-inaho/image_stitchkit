#include "internal.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <numeric>
#include <set>

namespace stitchkit::detail {
namespace {
using Terms = std::vector<std::pair<int, double>>;
void add_basis(Terms& row, const Mesh& mesh, int offset, Point p, int component, double scale) {
    const auto b = mesh.basis(p);
    for (int k = 0; k < 4; ++k)
        row.emplace_back(offset + 2 * b.indices[k] + component, scale * b.weights[k]);
}
std::vector<int> neighbours(const Mesh& m, int index) {
    std::vector<int> out;
    const int x = index % m.columns, y = index / m.columns;
    if (x > 0)
        out.push_back(index - 1);
    if (x < m.columns - 1)
        out.push_back(index + 1);
    if (y > 0)
        out.push_back(index - m.columns);
    if (y < m.rows - 1)
        out.push_back(index + m.columns);
    return out;
}
double support_distance(Point p, const std::vector<Point>& support, Size size) {
    double best = std::hypot(static_cast<double>(size.width), static_cast<double>(size.height));
    const double diagonal = best;
    for (const auto q : support)
        best = std::min(best, std::sqrt((p - q).squared_norm()));
    return best / diagonal;
}
void add_similarity(LeastSquaresProblem& system, const Mesh& mesh, int offset,
                    const Homography& prior, const std::vector<Point>& support, const Options& o) {
    for (int a = 0; a < static_cast<int>(mesh.original.size()); ++a) {
        for (int b : neighbours(mesh, a)) {
            if (b <= a)
                continue;
            std::set<int> indices;
            for (int k : neighbours(mesh, a))
                if (k != a)
                    indices.insert(k);
            for (int k : neighbours(mesh, b))
                if (k != a)
                    indices.insert(k);
            const std::vector<int> ns(indices.begin(), indices.end());
            Eigen::MatrixXd e(2 * static_cast<int>(ns.size()), 2);
            for (int k = 0; k < static_cast<int>(ns.size()); ++k) {
                const Point d = mesh.original[ns[k]] - mesh.original[a];
                e.row(2 * k) << d.x, -d.y;
                e.row(2 * k + 1) << d.y, d.x;
            }
            // The two columns are orthogonal for an orientation-preserving similarity.
            const Eigen::MatrixXd g = (e.transpose() * e).ldlt().solve(e.transpose());
            const Point d = mesh.original[b] - mesh.original[a];
            Eigen::Matrix2d main;
            main << d.x, -d.y, d.y, d.x;
            const Eigen::MatrixXd local = main * g;
            const double weight =
                o.global_beta +
                o.global_gamma * support_distance((mesh.original[a] + mesh.original[b]) * .5,
                                                  support, mesh.image_size);
            const double desired[2] = {prior.values[0], prior.values[3]};
            for (int component = 0; component < 2; ++component) {
                Terms lr{{offset + 2 * b + component, 1}, {offset + 2 * a + component, -1}}, gr;
                for (int k = 0; k < static_cast<int>(ns.size()); ++k)
                    for (int dim = 0; dim < 2; ++dim) {
                        const double lc = local(component, 2 * k + dim),
                                     gc = g(component, 2 * k + dim);
                        lr.emplace_back(offset + 2 * ns[k] + dim, -lc);
                        lr.emplace_back(offset + 2 * a + dim, lc);
                        gr.emplace_back(offset + 2 * ns[k] + dim, gc);
                        gr.emplace_back(offset + 2 * a + dim, -gc);
                    }
                system.add_row(lr, 0, o.local_similarity_weight);
                system.add_row(gr, desired[component], weight);
            }
        }
    }
}
void add_structures(LeastSquaresProblem& system, const Mesh& mesh, int offset,
                    const std::vector<Structure>& structures, const Options& o,
                    Diagnostics& diagnostic) {
    for (const auto& s : structures) {
        if (s.points.size() < 3 || !std::isfinite(s.confidence) || s.confidence <= 0)
            continue;
        // Farthest sampled pair also gives a stable baseline for closed curved contours.
        std::size_t ia = 0, ib = 0;
        double longest = 0;
        for (std::size_t a = 0; a < s.points.size(); ++a)
            for (std::size_t b = a + 1; b < s.points.size(); ++b) {
                const double length = (s.points[b] - s.points[a]).squared_norm();
                if (length > longest) {
                    longest = length;
                    ia = a;
                    ib = b;
                }
            }
        if (longest < 64)
            continue;
        const Point a = s.points[ia], b = s.points[ib], axis = b - a;
        const double factor = o.structure_weight * std::sqrt(std::clamp(s.confidence, 0.0, 1.0));
        if (factor == 0)
            continue;
        ++diagnostic.structures;
        for (std::size_t k = 0; k < s.points.size(); ++k) {
            if (k == ia || k == ib)
                continue;
            const Point p = s.points[k], delta = p - a;
            const double u = dot(delta, axis) / longest, v = cross(axis, delta) / longest;
            // p' = a' + u(b'-a') + v R90(b'-a'), two coupled scalar equations.
            Terms rx, ry;
            add_basis(rx, mesh, offset, p, 0, 1);
            add_basis(rx, mesh, offset, a, 0, -(1 - u));
            add_basis(rx, mesh, offset, b, 0, -u);
            add_basis(rx, mesh, offset, b, 1, v);
            add_basis(rx, mesh, offset, a, 1, -v);
            add_basis(ry, mesh, offset, p, 1, 1);
            add_basis(ry, mesh, offset, a, 1, -(1 - u));
            add_basis(ry, mesh, offset, b, 1, -u);
            add_basis(ry, mesh, offset, b, 0, -v);
            add_basis(ry, mesh, offset, a, 0, v);
            system.add_row(rx, 0, factor);
            system.add_row(ry, 0, factor);
            diagnostic.structure_equations += 2;
        }
    }
}
class MeshEstimator final : public IWarpEstimator {
    Method method_;
    std::shared_ptr<ILeastSquaresSolver> solver_;

  public:
    MeshEstimator(Method method, std::shared_ptr<ILeastSquaresSolver> solver)
        : method_(method), solver_(std::move(solver)) {
        if (!solver_)
            throw Error("Mesh estimator requires a solver");
    }
    WarpPlan estimate(const Registration& r, const std::vector<std::vector<Structure>>& structures,
                      const Options& o, Diagnostics& diagnostic) const override {
        if (r.sizes.size() < 2 || r.global_similarities.size() != r.sizes.size() ||
            r.global_homographies.size() != r.sizes.size())
            throw Error("Invalid registration for mesh optimization");
        std::vector<Mesh> meshes;
        std::vector<int> offsets;
        std::vector<std::vector<Point>> support(r.sizes.size());
        LeastSquaresProblem system;
        for (const auto size : r.sizes) {
            offsets.push_back(system.columns);
            meshes.push_back(Mesh::regular(size, o.grid_size));
            system.columns += 2 * static_cast<int>(meshes.back().original.size());
        }
        double before = 0;
        int count = 0;
        for (const auto& pair : r.pairs) {
            const auto si = pair.source_index, ti = pair.target_index;
            if (si >= meshes.size() || ti >= meshes.size())
                throw Error("Registration index out of bounds");
            auto all = pair.matches;
            if (o.use_apap) {
                auto dense = apap_matches(pair, meshes[si], r.sizes[ti], o);
                diagnostic.dense_matches += static_cast<int>(dense.size());
                all.insert(all.end(), dense.begin(), dense.end());
            }
            for (const auto& m : pair.matches) {
                support[si].push_back(m.source);
                support[ti].push_back(m.target);
                const Point a = r.global_homographies[si].apply(m.source),
                            b = r.global_homographies[ti].apply(m.target);
                before += (a - b).squared_norm();
                ++count;
            }
            for (const auto& m : all)
                for (int dim = 0; dim < 2; ++dim) {
                    Terms row;
                    add_basis(row, meshes[si], offsets[si], m.source, dim, 1);
                    add_basis(row, meshes[ti], offsets[ti], m.target, dim, -1);
                    system.add_row(row, 0, o.alignment_weight * std::sqrt(m.confidence));
                }
        }
        // One translation anchor removes the two remaining gauge freedoms. Scale and
        // rotation are set by the global similarity prior, not by freezing an entire image.
        system.add_row({{0, 1}}, 0, 1000);
        system.add_row({{1, 1}}, 0, 1000);
        for (std::size_t i = 0; i < meshes.size(); ++i) {
            add_similarity(system, meshes[i], offsets[i], r.global_similarities[i], support[i], o);
            if (method_ == Method::gesgsp && i < structures.size())
                add_structures(system, meshes[i], offsets[i], structures[i], o, diagnostic);
        }
        diagnostic.equations = static_cast<int>(system.rhs.size());
        diagnostic.unknowns = system.columns;
        const auto solution = solver_->solve(system);
        if (solution.values.size() != static_cast<std::size_t>(system.columns))
            throw Error("Solver returned incorrect vector size");
        diagnostic.solver_relative_residual = solution.relative_residual;
        for (std::size_t i = 0; i < meshes.size(); ++i) {
            for (std::size_t v = 0; v < meshes[i].original.size(); ++v)
                meshes[i].deformed[v] = {solution.values[offsets[i] + 2 * v],
                                         solution.values[offsets[i] + 2 * v + 1]};
            meshes[i].validate_orientation();
        }
        double after = 0;
        for (const auto& pair : r.pairs)
            for (const auto& m : pair.matches)
                after += (meshes[pair.source_index].map(m.source) -
                          meshes[pair.target_index].map(m.target))
                             .squared_norm();
        diagnostic.alignment_rmse_before = count ? std::sqrt(before / count) : 0;
        diagnostic.alignment_rmse_after = count ? std::sqrt(after / count) : 0;
        return maps_from_meshes(std::move(meshes), o);
    }
};
} // namespace
std::unique_ptr<IWarpEstimator> make_mesh_estimator(Method method,
                                                    std::shared_ptr<ILeastSquaresSolver> solver) {
    return std::make_unique<MeshEstimator>(method, std::move(solver));
}
} // namespace stitchkit::detail
