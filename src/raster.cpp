#include "internal.hpp"
#include <algorithm>
#include <limits>

namespace stitchkit::detail {
Canvas bounded_canvas(const std::vector<Point>& points, std::size_t max_pixels, double padding) {
    if (points.empty() || !std::isfinite(padding) || padding < 0)
        throw Error("Cannot bound an empty or invalid warp");
    Point low{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    Point high{-low.x, -low.y};
    for (const auto p : points) {
        if (!finite(p) || std::abs(p.x) > 1e8 || std::abs(p.y) > 1e8)
            throw Error("Unbounded or non-finite warp");
        low.x = std::min(low.x, p.x);
        low.y = std::min(low.y, p.y);
        high.x = std::max(high.x, p.x);
        high.y = std::max(high.y, p.y);
    }
    low = {std::floor(low.x - padding + 1e-7), std::floor(low.y - padding + 1e-7)};
    high = {std::ceil(high.x + padding - 1e-7), std::ceil(high.y + padding - 1e-7)};
    const double width = high.x - low.x + 1, height = high.y - low.y + 1;
    if (width < 2 || height < 2 || width > 32768 || height > 32768 ||
        width * height > static_cast<double>(max_pixels))
        throw Error("Output canvas exceeds the configured size limit");
    return {static_cast<int>(width), static_cast<int>(height), low};
}
namespace {
Point bilinear(const std::array<Point, 4>& p, double u, double v) {
    return p[0] * ((1 - u) * (1 - v)) + p[1] * (u * (1 - v)) + p[2] * (u * v) +
           p[3] * ((1 - u) * v);
}
bool inverse_quad(const std::array<Point, 4>& p, Point q, double& u, double& v) {
    const Point a = p[1] - p[0], b = p[3] - p[0], c = p[0] - p[1] + p[2] - p[3], d = q - p[0];
    const double det = cross(a, b);
    if (std::abs(det) < 1e-10)
        return false;
    u = cross(d, b) / det;
    v = cross(a, d) / det;
    for (int iteration = 0; iteration < 12; ++iteration) {
        const Point residual = p[0] + a * u + b * v + c * (u * v) - q, du = a + c * v,
                    dv = b + c * u;
        const double j = cross(du, dv);
        if (std::abs(j) < 1e-10)
            return false;
        const double su = cross(residual, dv) / j, sv = cross(du, residual) / j;
        u -= su;
        v -= sv;
        if (std::abs(su) + std::abs(sv) < 1e-9)
            break;
    }
    return std::isfinite(u) && std::isfinite(v) && u >= -1e-6 && u <= 1 + 1e-6 && v >= -1e-6 &&
           v <= 1 + 1e-6 && (bilinear(p, u, v) - q).squared_norm() < 1e-6;
}
} // namespace
WarpPlan maps_from_meshes(std::vector<Mesh> meshes, const Options& o) {
    std::vector<Point> points;
    for (const auto& m : meshes)
        points.insert(points.end(), m.deformed.begin(), m.deformed.end());
    WarpPlan plan;
    plan.canvas = bounded_canvas(points, o.max_canvas_pixels);
    plan.meshes = std::move(meshes);
    const std::size_t count = static_cast<std::size_t>(plan.canvas.width) * plan.canvas.height;
    if (count * plan.meshes.size() > 64'000'000)
        throw Error("Sampling maps exceed the 64M-sample working-memory limit");
    for (const auto& m : plan.meshes) {
        SamplingMap map;
        map.x.assign(count, -1);
        map.y.assign(count, -1);
        for (int y = 0; y < m.rows - 1; ++y)
            for (int x = 0; x < m.columns - 1; ++x) {
                const int a = y * m.columns + x;
                const std::array<int, 4> ids{a, a + 1, a + m.columns + 1, a + m.columns};
                std::array<Point, 4> dst, src;
                for (int k = 0; k < 4; ++k) {
                    dst[k] = m.deformed[ids[k]] - plan.canvas.origin;
                    src[k] = m.original[ids[k]];
                }
                double minx = dst[0].x, maxx = minx, miny = dst[0].y, maxy = miny;
                for (auto p : dst) {
                    minx = std::min(minx, p.x);
                    maxx = std::max(maxx, p.x);
                    miny = std::min(miny, p.y);
                    maxy = std::max(maxy, p.y);
                }
                const int left = std::max(0, static_cast<int>(std::ceil(minx - 1e-6))),
                          right = std::min(plan.canvas.width - 1,
                                           static_cast<int>(std::floor(maxx + 1e-6)));
                const int top = std::max(0, static_cast<int>(std::ceil(miny - 1e-6))),
                          bottom = std::min(plan.canvas.height - 1,
                                            static_cast<int>(std::floor(maxy + 1e-6)));
                for (int py = top; py <= bottom; ++py)
                    for (int px = left; px <= right; ++px) {
                        double u = 0, v = 0;
                        if (!inverse_quad(dst, {static_cast<double>(px), static_cast<double>(py)},
                                          u, v))
                            continue;
                        const Point q = bilinear(src, std::clamp(u, 0., 1.), std::clamp(v, 0., 1.));
                        const auto index = static_cast<std::size_t>(py) * plan.canvas.width + px;
                        map.x[index] = static_cast<float>(q.x);
                        map.y[index] = static_cast<float>(q.y);
                    }
            }
        plan.maps.push_back(std::move(map));
    }
    return plan;
}
void validate_plan(const std::vector<Image>& images, const WarpPlan& plan, const Options& o) {
    o.validate();
    if (images.empty() || plan.canvas.width < 2 || plan.canvas.height < 2 || plan.maps.size() != images.size() ||
        !finite(plan.canvas.origin))
        throw Error("Invalid warp plan");
    const auto n = static_cast<std::size_t>(plan.canvas.width) * plan.canvas.height;
    if (n > o.max_canvas_pixels || n * images.size() > 64'000'000)
        throw Error("Warp plan exceeds memory limits");
    for (std::size_t i = 0; i < images.size(); ++i) {
        images[i].validate();
        if (plan.maps[i].x.size() != n || plan.maps[i].y.size() != n)
            throw Error("Sampling map size mismatch");
    }
}
} // namespace stitchkit::detail
namespace stitchkit {
namespace {
class CpuRenderer final : public IRenderer {
  public:
    std::string device_name() const override { return "cpu"; }
    Result render(const std::vector<Image>& images, const WarpPlan& plan,
                  const Options& o) const override {
        detail::validate_plan(images, plan, o);
        Result result;
        result.canvas = plan.canvas;
        result.image = Image(plan.canvas.width, plan.canvas.height);
        result.mask.assign(result.image.pixel_count(), 0);
        for (std::size_t p = 0; p < result.image.pixel_count(); ++p) {
            double values[3]{0, 0, 0}, weight_sum = 0;
            for (std::size_t i = 0; i < images.size(); ++i) {
                const double x = plan.maps[i].x[p], y = plan.maps[i].y[p];
                const auto& image = images[i];
                if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 ||
                    x > image.width - 1 || y > image.height - 1)
                    continue;
                const double distance = std::min({x + 1, y + 1, image.width - x, image.height - y});
                const double weight = std::min(1., distance / o.feather_width);
                const auto rgb = detail::sample(image, x, y);
                for (int c = 0; c < 3; ++c)
                    values[c] += weight * rgb[c];
                weight_sum += weight;
            }
            if (weight_sum > 0) {
                result.mask[p] = 255;
                for (int c = 0; c < 3; ++c)
                    result.image.pixels[3 * p + c] = static_cast<std::uint8_t>(
                        std::clamp(std::lround(values[c] / weight_sum), 0L, 255L));
            }
        }
        return result;
    }
};
} // namespace
bool cuda_compiled() noexcept {
#ifdef STITCHKIT_WITH_CUDA
    return true;
#else
    return false;
#endif
}
#ifndef STITCHKIT_WITH_CUDA
bool cuda_available() noexcept {
    return false;
}
#endif
std::unique_ptr<IRenderer> make_renderer(Device device, std::vector<std::string>& warnings) {
    (void)to_string(device); // Reject invalid enum values instead of silently selecting auto.
    if (device == Device::cpu)
        return std::make_unique<CpuRenderer>();
#ifdef STITCHKIT_WITH_CUDA
    if (cuda_available())
        return detail::make_cuda_renderer();
#endif
    if (device == Device::cuda)
        throw Error("CUDA explicitly requested, but no usable CUDA renderer/device is available");
    warnings.push_back("CUDA renderer unavailable; auto selected CPU. Geometry and feature "
                       "extraction remain CPU.");
    return std::make_unique<CpuRenderer>();
}
} // namespace stitchkit
