#include "internal.hpp"
#include <Eigen/LU>
#include <algorithm>
#include <limits>

namespace stitchkit {
namespace {
std::size_t checked_count(int w, int h) {
    if (w < 2 || h < 2 || w > 32768 || h > 32768)
        throw Error("Image dimensions must be between 2 and 32768");
    const auto count = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
    if (count > 64'000'000)
        throw Error("Image exceeds the 64 megapixel input limit");
    return count;
}
void positive(double v, const char* name) {
    if (!std::isfinite(v) || v <= 0)
        throw Error(std::string(name) + " must be finite and positive");
}
} // namespace
Image::Image(int w, int h, std::uint8_t value)
    : width(w), height(h), pixels(checked_count(w, h) * 3, value) {
}
std::size_t Image::pixel_count() const {
    return checked_count(width, height);
}
void Image::validate() const {
    if (pixels.size() != pixel_count() * 3)
        throw Error("Invalid RGB image buffer size");
}
const char* to_string(Method m) {
    switch (m) {
    case Method::rew:
        return "rew";
    case Method::niswgsp:
        return "niswgsp";
    case Method::gesgsp:
        return "gesgsp";
    }
    throw Error("Invalid method enum");
}
const char* to_string(Device d) {
    switch (d) {
    case Device::cpu:
        return "cpu";
    case Device::cuda:
        return "cuda";
    case Device::automatic:
        return "auto";
    }
    throw Error("Invalid device enum");
}
Method parse_method(const std::string& s) {
    if (s == "rew")
        return Method::rew;
    if (s == "niswgsp")
        return Method::niswgsp;
    if (s == "gesgsp")
        return Method::gesgsp;
    throw Error("Unknown method '" + s + "'; expected rew, niswgsp or gesgsp");
}
Device parse_device(const std::string& s) {
    if (s == "cpu")
        return Device::cpu;
    if (s == "cuda")
        return Device::cuda;
    if (s == "auto")
        return Device::automatic;
    throw Error("Unknown device '" + s + "'; expected cpu, cuda or auto");
}
void Options::validate() const {
    (void)to_string(method);
    (void)to_string(device);
    (void)to_string(lineae_device);
    if (structures != StructureMode::edges && structures != StructureMode::lineae &&
        structures != StructureMode::none)
        throw Error("Invalid structure mode");
    if (grid_size < 8 || grid_size > 256)
        throw Error("grid_size must be in [8,256]");
    if (max_features < 32 || max_features > 12000)
        throw Error("max_features must be in [32,12000]");
    if (!std::isfinite(ratio_threshold) || ratio_threshold <= 0 || ratio_threshold >= 1)
        throw Error("ratio_threshold must be in (0,1)");
    if (minimum_matches < 6 || minimum_matches > max_features)
        throw Error("Invalid minimum_matches");
    if (ransac_iterations < 32 || ransac_iterations > 100000)
        throw Error("Invalid ransac_iterations");
    if (max_tps_controls < 6 || max_tps_controls > 2000)
        throw Error("Invalid max_tps_controls");
    if (max_structures < 1 || max_structures > 1000)
        throw Error("Invalid max_structures");
    if (max_canvas_pixels < 100 || max_canvas_pixels > 64'000'000)
        throw Error("Invalid max_canvas_pixels");
    positive(ransac_threshold, "ransac_threshold");
    positive(apap_sigma, "apap_sigma");
    positive(alignment_weight, "alignment_weight");
    positive(local_similarity_weight, "local_similarity_weight");
    positive(global_beta, "global_beta");
    positive(global_gamma, "global_gamma");
    if (!std::isfinite(structure_weight) || structure_weight < 0)
        throw Error("structure_weight must be nonnegative");
    positive(tps_lambda, "tps_lambda");
    positive(feather_width, "feather_width");
    if (!std::isfinite(lineae_threshold) || lineae_threshold < 0 || lineae_threshold > 1)
        throw Error("lineae_threshold must be in [0,1]");
    if (structures == StructureMode::lineae && (lineae_model.empty() || lineae_variant.empty()))
        throw Error("LINEAE requires both --lineae-model and --lineae-variant");
    if (structures == StructureMode::lineae && method != Method::gesgsp)
        throw Error("LINEAE structure constraints are only used by gesgsp");
}
Point Homography::apply(Point p) const {
    const double z = values[6] * p.x + values[7] * p.y + values[8];
    if (!finite(p) || !std::isfinite(z) || std::abs(z) < 1e-10)
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    return {(values[0] * p.x + values[1] * p.y + values[2]) / z,
            (values[3] * p.x + values[4] * p.y + values[5]) / z};
}
Homography Homography::inverse() const {
    const auto m = detail::matrix(*this);
    Eigen::FullPivLU<Eigen::Matrix3d> lu(m);
    if (!m.allFinite() || !lu.isInvertible())
        throw Error("Singular homography");
    return detail::homography(lu.inverse());
}
Homography compose(const Homography& a, const Homography& b) {
    return detail::homography(detail::matrix(a) * detail::matrix(b));
}
Mesh Mesh::regular(Size size, int cell) {
    (void)checked_count(size.width, size.height);
    if (cell < 1)
        throw Error("Invalid mesh cell size");
    Mesh m;
    m.image_size = size;
    m.columns = std::max(1, 1 + (size.width - 2) / cell) + 1;
    m.rows = std::max(1, 1 + (size.height - 2) / cell) + 1;
    if (static_cast<std::size_t>(m.columns) * m.rows > 100'000)
        throw Error("Mesh is too large; increase grid_size");
    for (int y = 0; y < m.rows; ++y)
        for (int x = 0; x < m.columns; ++x)
            m.original.push_back({double(x) * (size.width - 1) / (m.columns - 1),
                                  double(y) * (size.height - 1) / (m.rows - 1)});
    m.deformed = m.original;
    return m;
}
Basis Mesh::basis(Point p) const {
    if (!finite(p) || columns < 2 || rows < 2 || image_size.width < 2 || image_size.height < 2)
        throw Error("Invalid mesh interpolation input");
    if (p.x < -1e-5 || p.y < -1e-5 || p.x > image_size.width - 1 + 1e-5 ||
        p.y > image_size.height - 1 + 1e-5)
        throw Error("Interpolation point is outside its image");
    const double gx =
        std::clamp(p.x, 0.0, double(image_size.width - 1)) * (columns - 1) / (image_size.width - 1);
    const double gy =
        std::clamp(p.y, 0.0, double(image_size.height - 1)) * (rows - 1) / (image_size.height - 1);
    const int x = std::min(static_cast<int>(gx), columns - 2),
              y = std::min(static_cast<int>(gy), rows - 2);
    const double u = gx - x, v = gy - y;
    const int i = y * columns + x;
    return {{i, i + 1, i + columns + 1, i + columns},
            {(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v}};
}
Point Mesh::map(Point p) const {
    const auto b = basis(p);
    Point out;
    for (int i = 0; i < 4; ++i)
        out = out + deformed.at(b.indices[i]) * b.weights[i];
    return out;
}
void Mesh::validate_orientation() const {
    if (columns < 2 || rows < 2 || deformed.size() != static_cast<std::size_t>(columns) * rows)
        throw Error("Invalid deformed mesh");
    for (Point p : deformed)
        if (!finite(p))
            throw Error("Non-finite mesh vertex");
    for (int y = 0; y < rows - 1; ++y)
        for (int x = 0; x < columns - 1; ++x) {
            const int i = y * columns + x;
            const Point a = deformed[i], b = deformed[i + 1], c = deformed[i + columns + 1],
                        d = deformed[i + columns];
            // The bilinear Jacobian is affine in (u,v); positive at all corners implies no fold.
            if (cross(b - a, d - a) <= 1e-7 || cross(b - a, c - b) <= 1e-7 ||
                cross(c - d, c - b) <= 1e-7 || cross(c - d, d - a) <= 1e-7)
                throw Error("Folded or degenerate mesh; reduce alignment/structure weights or "
                            "increase grid size");
        }
}
} // namespace stitchkit

namespace stitchkit::detail {
Matrix3 matrix(const Homography& h) {
    Matrix3 m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            m(r, c) = h.values[3 * r + c];
    return m;
}
Homography homography(const Matrix3& input) {
    if (!input.allFinite())
        throw Error("Non-finite transformation");
    Matrix3 m = input;
    const double scale = std::abs(m(2, 2)) > 1e-12 ? m(2, 2) : m.norm();
    if (std::abs(scale) < 1e-12)
        throw Error("Zero transformation");
    m /= scale;
    Homography h;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            h.values[3 * r + c] = m(r, c);
    return h;
}
std::array<float, 3> sample(const Image& im, double x, double y) {
    if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 || x > im.width - 1 ||
        y > im.height - 1)
        return {0, 0, 0};
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const int x1 = std::min(x0 + 1, im.width - 1), y1 = std::min(y0 + 1, im.height - 1);
    const float u = static_cast<float>(x - x0), v = static_cast<float>(y - y0);
    std::array<float, 3> out{};
    for (int c = 0; c < 3; ++c) {
        const float a = im.pixels[(static_cast<std::size_t>(y0) * im.width + x0) * 3 + c];
        const float b = im.pixels[(static_cast<std::size_t>(y0) * im.width + x1) * 3 + c];
        const float d = im.pixels[(static_cast<std::size_t>(y1) * im.width + x0) * 3 + c];
        const float e = im.pixels[(static_cast<std::size_t>(y1) * im.width + x1) * 3 + c];
        out[c] = (a * (1 - u) + b * u) * (1 - v) + (d * (1 - u) + e * u) * v;
    }
    return out;
}
std::vector<float> grayscale(const Image& im) {
    im.validate();
    std::vector<float> result(im.pixel_count());
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = 0.299f * im.pixels[i * 3] + 0.587f * im.pixels[i * 3 + 1] +
                    0.114f * im.pixels[i * 3 + 2];
    return result;
}
} // namespace stitchkit::detail
