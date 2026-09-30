#include "internal.hpp"
#include <Eigen/SparseQR>
#include <algorithm>

namespace stitchkit {
void LeastSquaresProblem::add_row(const std::vector<std::pair<int, double>>& terms, double target,
                                  double weight) {
    if (columns <= 0 || !std::isfinite(target) || !std::isfinite(weight) || weight < 0)
        throw Error("Invalid least-squares row");
    if (weight == 0)
        return;
    if (rhs.size() >= 2'000'000)
        throw Error("Too many least-squares equations");
    const double weighted_target = target * weight;
    if (!std::isfinite(weighted_target))
        throw Error("Weighted least-squares target overflow");
    const int row = static_cast<int>(rhs.size());
    std::vector<Coefficient> staged;
    staged.reserve(terms.size());
    for (const auto& term : terms) {
        const double value = term.second * weight;
        if (term.first < 0 || term.first >= columns || !std::isfinite(value))
            throw Error("Invalid or overflowing least-squares coefficient");
        if (value != 0)
            staged.push_back({row, term.first, value});
    }
    // Allocate before committing either vector. A rejected row cannot poison later rows.
    rhs.reserve(rhs.size() + 1);
    coefficients.reserve(coefficients.size() + staged.size());
    coefficients.insert(coefficients.end(), staged.begin(), staged.end());
    rhs.push_back(weighted_target);
}
namespace {
class EigenSolver final : public ILeastSquaresSolver {
  public:
    LeastSquaresSolution solve(const LeastSquaresProblem& p) const override {
        if (p.columns <= 0 || p.rhs.empty() || p.rhs.size() < static_cast<std::size_t>(p.columns))
            throw Error("Underdetermined or empty least-squares system");
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(p.coefficients.size());
        for (const auto& v : p.coefficients) {
            if (v.row < 0 || v.row >= static_cast<int>(p.rhs.size()) || v.column < 0 ||
                v.column >= p.columns || !std::isfinite(v.value))
                throw Error("Invalid sparse matrix entry");
            entries.emplace_back(v.row, v.column, v.value);
        }
        Eigen::SparseMatrix<double> a(static_cast<int>(p.rhs.size()), p.columns);
        a.setFromTriplets(entries.begin(), entries.end());
        a.makeCompressed();
        const Eigen::Map<const Eigen::VectorXd> b(p.rhs.data(),
                                                  static_cast<Eigen::Index>(p.rhs.size()));
        if (!b.allFinite())
            throw Error("Non-finite right-hand side");
        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
        qr.compute(a);
        if (qr.info() != Eigen::Success || qr.rank() != p.columns)
            throw Error("Singular mesh system");
        const Eigen::VectorXd x = qr.solve(b);
        if (qr.info() != Eigen::Success || !x.allFinite())
            throw Error("Sparse least-squares solution failed");
        LeastSquaresSolution out;
        out.values.assign(x.data(), x.data() + x.size());
        out.relative_residual = (a * x - b).norm() / std::max(1.0, b.norm());
        // A large data residual is possible for an inconsistent scene. The normal-equation
        // residual verifies the numerical solve, not the suitability of the image model.
        const double stationarity =
            (a.transpose() * (a * x - b)).norm() / std::max(1.0, (a.transpose() * b).norm());
        if (!std::isfinite(out.relative_residual) || stationarity > 1e-6)
            throw Error("Mesh solve failed stationarity check");
        return out;
    }
};
} // namespace
std::shared_ptr<ILeastSquaresSolver> make_eigen_solver() {
    return std::make_shared<EigenSolver>();
}
} // namespace stitchkit
