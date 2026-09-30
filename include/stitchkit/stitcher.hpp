#pragma once
#include "stitchkit/interfaces.hpp"

namespace stitchkit {

// An immutable, owning snapshot. No public constructor or mutable access to its images
// or correspondences; sharing a snapshot never aliases a caller's input buffers.
class PreparedScene {
  public:
    std::size_t image_count() const;
    double registration_ms() const;
    int feature_extractions() const;
  private:
    struct State;
    std::shared_ptr<const State> state_;
    explicit PreparedScene(std::shared_ptr<const State> state);
    friend class Stitcher;
};

class Stitcher {
  public:
    explicit Stitcher(Options options = {});
    Stitcher(Options options, std::unique_ptr<IFeatureMatcher> matcher,
             std::unique_ptr<IStructureDetector> detector,
             std::shared_ptr<ILeastSquaresSolver> solver,
             std::shared_ptr<IRenderer> renderer = {});
    // By-value input: lvalues are copied; owned temporary images can be moved in.
    Result stitch(std::vector<Image> images) const;
    PreparedScene prepare(std::vector<Image> images) const;
    Result compose(const PreparedScene& scene) const;
  private:
    Options options_;
    std::unique_ptr<IFeatureMatcher> matcher_;
    std::unique_ptr<IStructureDetector> detector_;
    std::shared_ptr<ILeastSquaresSolver> solver_;
    std::shared_ptr<IRenderer> renderer_;
    Result compose_impl(const PreparedScene& scene, IRenderer& renderer,
                        std::vector<std::string> warnings, bool reused) const;
};

Image read_image(const std::filesystem::path& path);
void write_png(const std::filesystem::path& path, const Image& image,
               const std::vector<std::uint8_t>& mask = {});
std::string report_json(const Result& result);
void write_report(const std::filesystem::path& path, const Result& result);
Image resize_image(const Image& input, Size target);
Image limit_size(const Image& input, int max_side);
std::string version();

} // namespace stitchkit
