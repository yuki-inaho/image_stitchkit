#pragma once
#include "stitchkit/interfaces.hpp"
#include <Eigen/Core>
#include <functional>

namespace stitchkit::detail {
using Matrix3 = Eigen::Matrix3d;
Matrix3 matrix(const Homography& h);
Homography homography(const Matrix3& m);
Homography fit_homography(const std::vector<Match>& matches,
                          const std::vector<double>& weights = {});
Homography fit_similarity(const std::vector<Match>& matches);
PairRegistration robust_register(std::vector<Match> matches, const Options& options);
std::vector<Match> apap_matches(const PairRegistration& pair, const Mesh& source, Size target,
                                const Options& options);
WarpPlan maps_from_meshes(std::vector<Mesh> meshes, const Options& options);
Canvas bounded_canvas(const std::vector<Point>& points, std::size_t max_pixels,
                      double padding = 0.0);
std::array<float, 3> sample(const Image& image, double x, double y);
std::vector<float> grayscale(const Image& image);
std::unique_ptr<IWarpEstimator> make_rew_estimator();
std::unique_ptr<IWarpEstimator> make_mesh_estimator(Method method,
                                                    std::shared_ptr<ILeastSquaresSolver> solver);
void validate_plan(const std::vector<Image>& images, const WarpPlan& plan, const Options& options);
#ifdef STITCHKIT_WITH_CUDA
std::unique_ptr<IRenderer> make_cuda_renderer();
#endif
#ifdef STITCHKIT_WITH_LINEAE
std::unique_ptr<IStructureDetector> make_onnx_lineae(const Options& options);
#endif
} // namespace stitchkit::detail
