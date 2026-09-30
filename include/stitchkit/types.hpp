#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stitchkit {

struct Point {
    double x = 0.0, y = 0.0;
    Point operator+(Point p) const { return {x + p.x, y + p.y}; }
    Point operator-(Point p) const { return {x - p.x, y - p.y}; }
    Point operator*(double s) const { return {x * s, y * s}; }
    Point operator/(double s) const { return {x / s, y / s}; }
    double squared_norm() const { return x * x + y * y; }
};
inline double dot(Point a, Point b) {
    return a.x * b.x + a.y * b.y;
}
inline double cross(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}
inline bool finite(Point p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

struct Size {
    int width = 0, height = 0;
};
struct Image {
    int width = 0, height = 0;
    std::vector<std::uint8_t> pixels; // Interleaved RGB, row-major; never BGR.
    Image() = default;
    Image(int w, int h, std::uint8_t value = 0);
    void validate() const;
    std::size_t pixel_count() const;
    Size size() const { return {width, height}; }
};

class Error : public std::runtime_error {
  public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

enum class Method { rew, niswgsp, gesgsp };
enum class Device { cpu, cuda, automatic };
enum class StructureMode { edges, lineae, none };
const char* to_string(Method value);
const char* to_string(Device value);
Method parse_method(const std::string& value);
Device parse_device(const std::string& value);

struct Options {
    Method method = Method::gesgsp;
    Device device = Device::cpu; // Explicit CPU is the reproducible default.
    StructureMode structures = StructureMode::edges;
    int grid_size = 32;
    int max_features = 2500;
    double ratio_threshold = 0.78;
    double ransac_threshold = 3.0;
    int ransac_iterations = 1200;
    int minimum_matches = 10;
    std::uint32_t seed = 42;
    bool use_apap = true;
    double apap_sigma = 80.0;
    double alignment_weight = 1.0;
    double local_similarity_weight = 0.75;
    double global_beta = 6.0;
    double global_gamma = 20.0;
    double structure_weight = 3.0;
    double tps_lambda = 0.001;
    int max_tps_controls = 384;
    int max_structures = 100;
    double feather_width = 32.0;
    std::size_t max_canvas_pixels = 32'000'000;
    std::filesystem::path lineae_model;
    std::string lineae_variant; // Required for LINEAE, no filename guessing.
    Device lineae_device = Device::cpu;
    double lineae_threshold = 0.3;
    void validate() const;
};

struct Match {
    Point source, target;
    double confidence = 1.0;
};
struct Homography {
    std::array<double, 9> values{1, 0, 0, 0, 1, 0, 0, 0, 1};
    Point apply(Point p) const;
    Homography inverse() const;
};
Homography compose(const Homography& a, const Homography& b);

struct PairRegistration {
    std::size_t target_index = 0, source_index = 1;
    Homography homography; // Source -> target.
    Homography similarity; // Source -> target, orientation-preserving 2D similarity.
    std::vector<Match> matches;
    int tentative_matches = 0;
    int dominant_inliers = 0;
    double reprojection_rmse = 0.0;
};
struct Registration {
    std::vector<Size> sizes;
    std::vector<PairRegistration> pairs; // Adjacent, ordered, connected chain.
    std::vector<Homography> global_homographies;
    std::vector<Homography> global_similarities;
};
struct Structure {
    std::vector<Point> points; // Ordered open or closed edge samples.
    double confidence = 1.0;
};
struct Basis {
    std::array<int, 4> indices{};
    std::array<double, 4> weights{};
};
struct Mesh {
    int columns = 0, rows = 0; // Vertex counts, not cell counts.
    Size image_size;
    std::vector<Point> original;
    std::vector<Point> deformed;
    static Mesh regular(Size size, int cell_size);
    Basis basis(Point p) const;
    Point map(Point p) const;
    void validate_orientation() const;
};

struct Canvas {
    int width = 0, height = 0;
    Point origin; // Position of output pixel (0,0) in panorama coordinates.
};
struct SamplingMap {
    std::vector<float> x, y; // Negative coordinates denote invalid samples.
};
struct WarpPlan {
    Canvas canvas;
    std::vector<SamplingMap> maps;
    std::vector<Mesh> meshes; // Empty for inverse TPS warps.
};
struct Diagnostics {
    Options options;
    std::vector<Size> input_sizes;
    std::string method;
    std::string requested_device;
    std::string actual_device;
    std::string line_device;
    bool registration_reused = false;
    int feature_extractions = -1;
    double prepared_registration_ms = 0.0; // Historical cost, not part of this call.
    int feature_matches = 0;
    int dense_matches = 0;
    int structures = 0;
    int structure_equations = 0;
    int tps_controls = 0;
    int equations = 0;
    int unknowns = 0;
    double alignment_rmse_before = 0.0;
    double alignment_rmse_after = 0.0;
    double solver_relative_residual = 0.0;
    double registration_ms = 0.0;
    double structure_ms = 0.0;
    double geometry_ms = 0.0;
    double render_ms = 0.0;
    double total_ms = 0.0;
    std::vector<std::string> warnings;
};
struct Result {
    Image image;
    std::vector<std::uint8_t> mask; // 255 means a valid pixel, even when RGB is black.
    Canvas canvas;
    Diagnostics diagnostics;
};

} // namespace stitchkit
