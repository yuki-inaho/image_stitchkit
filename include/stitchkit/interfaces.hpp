#pragma once
#include "stitchkit/types.hpp"
#include <memory>
#include <utility>

namespace stitchkit {

struct MatchBatch {
    std::vector<PairRegistration> pairs;
    int feature_extractions = -1; // Unknown for third-party matchers.
};

class IFeatureMatcher {
  public:
    virtual ~IFeatureMatcher() = default;
    virtual PairRegistration match(const Image& source, const Image& target,
                                   const Options& options) const = 0;
    // Default adapter preserves existing injected pair matchers.
    virtual MatchBatch match_sequence(const std::vector<Image>& images,
                                      const Options& options) const;
};
class IStructureDetector {
  public:
    virtual ~IStructureDetector() = default;
    virtual std::vector<Structure> detect(const Image& image, const Options& options) const = 0;
    virtual std::string device_name() const { return "cpu"; }
};
struct Coefficient {
    int row, column;
    double value;
};
struct LeastSquaresProblem {
    int columns = 0;
    std::vector<Coefficient> coefficients;
    std::vector<double> rhs;
    void add_row(const std::vector<std::pair<int, double>>& terms, double target,
                 double residual_weight = 1.0);
};
struct LeastSquaresSolution {
    std::vector<double> values;
    double relative_residual = 0.0;
};
class ILeastSquaresSolver {
  public:
    virtual ~ILeastSquaresSolver() = default;
    virtual LeastSquaresSolution solve(const LeastSquaresProblem& problem) const = 0;
};
class IWarpEstimator {
  public:
    virtual ~IWarpEstimator() = default;
    virtual WarpPlan estimate(const Registration& registration,
                              const std::vector<std::vector<Structure>>& structures,
                              const Options& options, Diagnostics& diagnostics) const = 0;
};
class IRenderer {
  public:
    virtual ~IRenderer() = default;
    virtual Result render(const std::vector<Image>& images, const WarpPlan& plan,
                          const Options& options) const = 0;
    virtual std::string device_name() const = 0;
};

std::unique_ptr<IFeatureMatcher> make_sift_matcher();
std::unique_ptr<IStructureDetector> make_edge_detector();
std::unique_ptr<IStructureDetector> make_lineae_detector(const Options& options);
std::shared_ptr<ILeastSquaresSolver> make_eigen_solver();
std::unique_ptr<IWarpEstimator>
make_warp_estimator(Method method,
                    std::shared_ptr<ILeastSquaresSolver> solver = make_eigen_solver());
std::unique_ptr<IRenderer> make_renderer(Device device, std::vector<std::string>& warnings);
bool cuda_available() noexcept;
bool cuda_compiled() noexcept;
bool lineae_compiled() noexcept;

} // namespace stitchkit
