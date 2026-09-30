#include <stitchkit/stitcher.hpp>
#include <stitchkit/lineae.hpp>
#include "internal.hpp"
#include "vl/imopv.h"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <limits>

using namespace stitchkit;
namespace {
std::filesystem::path assets, output_dir;
void require(bool condition, const std::string& message) {
    if (!condition)
        throw Error("TEST: " + message);
}
void near(double actual, double expected, double tolerance, const std::string& label) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance,
            label + ": actual=" + std::to_string(actual) +
                ", expected=" + std::to_string(expected));
}
template <class F> void throws(F function, const std::string& label) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception&) {
        caught = true;
    }
    require(caught, label + " must throw");
}
Registration known_registration(bool deform = false) {
    Registration r;
    r.sizes = {{128, 96}, {128, 96}};
    r.global_homographies.resize(2);
    r.global_similarities.resize(2);
    PairRegistration pair;
    for (int y = 12; y <= 84; y += 12)
        for (int x = 12; x <= 108; x += 12) {
            const double delta =
                deform ? 5 * std::exp(-std::pow((x - 60) / 35., 2) - std::pow((y - 45) / 30., 2))
                       : 0;
            pair.matches.push_back({{double(x), double(y)}, {x + delta, y + .25 * delta}, 1});
        }
    pair.dominant_inliers = static_cast<int>(pair.matches.size());
    r.pairs.push_back(pair);
    return r;
}
std::vector<Structure> known_structures() {
    std::vector<Structure> result;
    Structure line;
    for (int x = 8; x < 120; x += 8)
        line.points.push_back({double(x), 48});
    result.push_back(line);
    Structure circle;
    for (int k = 0; k < 18; ++k) {
        const double a = 2 * 3.14159265358979323846 * k / 18;
        circle.points.push_back({64 + 35 * std::cos(a), 48 + 30 * std::sin(a)});
    }
    result.push_back(circle);
    return result;
}
double structure_error(const Mesh& mesh, const std::vector<Structure>& structures) {
    double error = 0;
    for (const auto& s : structures) {
        std::size_t a = 0, b = 0;
        double longest = 0;
        for (std::size_t i = 0; i < s.points.size(); ++i)
            for (std::size_t j = i + 1; j < s.points.size(); ++j)
                if ((s.points[i] - s.points[j]).squared_norm() > longest) {
                    longest = (s.points[i] - s.points[j]).squared_norm();
                    a = i;
                    b = j;
                }
        const Point axis = s.points[b] - s.points[a], wa = mesh.map(s.points[a]),
                    wb = mesh.map(s.points[b]), wd = wb - wa;
        for (const auto p : s.points) {
            const Point q = p - s.points[a];
            const double u = dot(q, axis) / longest, v = cross(axis, q) / longest;
            error += (mesh.map(p) - wa - wd * u - Point{-wd.y, wd.x} * v).squared_norm();
        }
    }
    return error;
}
void test_convolution() {
    const float image[] = {1, 10, 2, 20, 3, 30};
    const float filter[] = {1, 2, 3};
    float output[6] = {};
    vl_imconvcol_vf(output, 2, image, 2, 3, 2, filter, -1, 1, 1, VL_PAD_BY_CONTINUITY);
    const double continuity[] = {7, 70, 10, 100, 15, 150};
    for (int k = 0; k < 6; ++k)
        near(output[k], continuity[k], 1e-6, "convolution continuity");
    vl_imconvcol_vf(output, 2, image, 2, 3, 2, filter, -1, 1, 1, VL_PAD_BY_ZERO);
    const double zero[] = {4, 40, 10, 100, 12, 120};
    for (int k = 0; k < 6; ++k)
        near(output[k], zero[k], 1e-6, "convolution zero padding");
    vl_imconvcol_vf(output, 2, image, 2, 3, 2, filter, -1, 1, 2,
                    VL_PAD_BY_CONTINUITY | VL_TRANSPOSE);
    const double transposed[] = {7, 15, 70, 150};
    for (int k = 0; k < 4; ++k)
        near(output[k], transposed[k], 1e-6, "convolution transpose/step");
}
void test_types() {
    throws([] { Image(-1, 20); }, "negative image size");
    throws([] { Image(32768, 32768); }, "image memory cap");
    auto image = Image(10, 12);
    image.pixels.pop_back();
    throws([&] { image.validate(); }, "short pixel buffer");
    auto o = Options{};
    o.ratio_threshold = 1;
    throws([&] { o.validate(); }, "ratio range");
    throws([] { parse_method("homography"); }, "unknown method");
    throws([] { parse_device("gpu-ish"); }, "unknown device");
    Homography h;
    h.values = {1.1, -.1, 20, .1, 1.1, -5, .0001, .0002, 1};
    const Point p{13, 41}, q = h.inverse().apply(h.apply(p));
    near(q.x, p.x, 1e-10, "homography inverse x");
    near(q.y, p.y, 1e-10, "homography inverse y");
    Homography singular;
    singular.values.fill(0);
    throws([&] { singular.inverse(); }, "singular homography");
}
void test_mesh() {
    auto mesh = Mesh::regular({107, 79}, 24);
    mesh.validate_orientation();
    for (const Point p :
         std::array<Point, 5>{{{0, 0}, {106, 78}, {33.5, 22.25}, {0, 78}, {106, 0}}}) {
        const auto basis = mesh.basis(p);
        near(std::accumulate(basis.weights.begin(), basis.weights.end(), 0.), 1, 1e-12,
             "partition of unity");
        near(mesh.map(p).x, p.x, 1e-10, "identity mesh x");
        near(mesh.map(p).y, p.y, 1e-10, "identity mesh y");
    }
    throws([&] { mesh.basis({-2, 3}); }, "mesh out of bounds");
    mesh.deformed[1] = mesh.deformed[0];
    throws([&] { mesh.validate_orientation(); }, "folded mesh");
}
void test_solver() {
    LeastSquaresProblem p;
    p.columns = 2;
    p.add_row({{0, 1}}, 2);
    p.add_row({{1, 1}}, 3);
    p.add_row({{0, 1}, {1, 1}}, 5);
    auto solution = make_eigen_solver()->solve(p);
    near(solution.values[0], 2, 1e-10, "least squares x");
    near(solution.values[1], 3, 1e-10, "least squares y");
    p.coefficients.clear();
    throws([&] { make_eigen_solver()->solve(p); }, "singular sparse solve");
    throws([&] { p.add_row({{2, 1}}, 0); }, "invalid sparse index");
}
void test_registration() {
    const double a = 1.05 * std::cos(.06), b = 1.05 * std::sin(.06);
    Homography truth;
    truth.values = {a, -b, 23, b, a, -7, 0, 0, 1};
    std::vector<Match> matches;
    for (int y = 10; y <= 90; y += 20)
        for (int x = 10; x <= 130; x += 20) {
            Point p{double(x), double(y)};
            matches.push_back({p, truth.apply(p), 1});
        }
    for (int k = 0; k < 8; ++k)
        matches.push_back({{double(k * 15 + 3), double(k * 7 + 2)},
                           {double(150 - k * 8), double(k * 13 + 7)},
                           1});
    Options o;
    o.ransac_threshold = .5;
    const auto result = detail::robust_register(matches, o);
    near((result.homography.apply({50, 50}) - truth.apply({50, 50})).squared_norm(), 0, 1e-12,
         "RANSAC known transform");
    require(result.dominant_inliers >= 35, "RANSAC retains true consensus");
    std::vector<Match> collinear;
    for (int k = 0; k < 15; ++k)
        collinear.push_back({{double(k), 0}, {double(k + 10), 0}, 1});
    throws([&] { detail::robust_register(collinear, o); }, "collinear correspondences");
}
void test_rew() {
    Options o;
    o.method = Method::rew;
    Diagnostics d;
    const auto r = known_registration();
    auto plan = make_warp_estimator(o.method)->estimate(r, {}, o, d);
    near(d.alignment_rmse_after, 0, 1e-9, "zero-parallax TPS residual");
    require(d.tps_controls > 10, "TPS fitted controls");
    require(plan.canvas.width == 128 && plan.canvas.height == 96, "identity canvas exact bounds");
    for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 128; ++x) {
            const auto i = std::size_t(y) * 128 + x;
            near(plan.maps[1].x[i], x, 1e-4, "TPS identity map x");
            near(plan.maps[1].y[i], y, 1e-4, "TPS identity map y");
        }
    Diagnostics bent;
    make_warp_estimator(o.method)->estimate(known_registration(true), {}, o, bent);
    require(bent.alignment_rmse_after < bent.alignment_rmse_before * .75,
            "elastic deformation reduces known residual");
    std::cout << "REW controlled residual: " << bent.alignment_rmse_before << " -> "
              << bent.alignment_rmse_after << '\n';
}
void test_geometric_energy() {
    Options o;
    o.grid_size = 16;
    o.use_apap = false;
    o.structure_weight = 20;
    const auto r = known_registration(true);
    const auto shapes = known_structures();
    Diagnostics dn, dg;
    const auto n = make_warp_estimator(Method::niswgsp)->estimate(r, {shapes, shapes}, o, dn);
    const auto g = make_warp_estimator(Method::gesgsp)->estimate(r, {shapes, shapes}, o, dg);
    const double en = structure_error(n.meshes[0], shapes) + structure_error(n.meshes[1], shapes);
    const double eg = structure_error(g.meshes[0], shapes) + structure_error(g.meshes[1], shapes);
    require(dn.structure_equations == 0 && dg.structure_equations > 0,
            "methods select different objectives");
    require(en > 1e-4 && eg < en * .2, "GES structure equations preserve lines and curves");
    std::cout << "Controlled structure residual sum: NIS=" << en << ", GES=" << eg << '\n';
}
void test_raster() {
    auto mesh = Mesh::regular({32, 24}, 40);
    mesh.deformed[1] = {36, 2};
    mesh.deformed[2] = {2, 28};
    mesh.deformed[3] = {40, 32};
    mesh.validate_orientation();
    Options o;
    const auto plan = detail::maps_from_meshes({mesh}, o);
    int valid = 0;
    for (int y = 0; y < plan.canvas.height; ++y)
        for (int x = 0; x < plan.canvas.width; ++x) {
            const auto p = std::size_t(y) * plan.canvas.width + x;
            if (plan.maps[0].x[p] < 0)
                continue;
            ++valid;
            const auto q = mesh.map({plan.maps[0].x[p], plan.maps[0].y[p]});
            near(q.x, x + plan.canvas.origin.x, 1e-4, "inverse bilinear raster x");
            near(q.y, y + plan.canvas.origin.y, 1e-4, "inverse bilinear raster y");
        }
    require(valid > 500, "quad raster has interior coverage");
    auto black = Image(32, 24, 0);
    std::vector<std::string> warnings;
    const auto result = make_renderer(Device::cpu, warnings)->render({black}, plan, o);
    require(std::count(result.mask.begin(), result.mask.end(), 255) == valid,
            "black pixels remain valid");
    require(std::count(result.image.pixels.begin(), result.image.pixels.end(), 0) ==
                static_cast<int>(result.image.pixels.size()),
            "black RGB unchanged");
    throws([&] { detail::bounded_canvas({{0, 0}, {1e9, 1}}, 1000); }, "runaway canvas");
}
void test_io() {
    const auto original = read_image(assets / "synthetic/view0.png");
    const auto path = output_dir / "io_roundtrip.png";
    write_png(path, original);
    const auto decoded = read_image(path);
    require(decoded.pixels == original.pixels, "PNG lossless roundtrip");
    auto jpeg = read_image(assets / "real/lib1.jpg");
    require(jpeg.width == 1000 && jpeg.height == 620, "JPEG RGB decode dimensions");
    auto resized = limit_size(original, 200);
    require(resized.width == 200 && resized.height == 140, "aspect ratio maintained");
    throws([&] { read_image(output_dir / "does_not_exist.png"); }, "missing input");
    {
        std::ofstream bad(output_dir / "corrupt.jpg", std::ios::binary);
        const char bytes[] = {char(0xff), char(0xd8), 0, 0, 0, 0, 0, 0};
        bad.write(bytes, 8);
    }
    throws([&] { read_image(output_dir / "corrupt.jpg"); }, "corrupt JPEG");
    {
        std::ofstream bad(output_dir / "corrupt.png", std::ios::binary);
        const unsigned char bytes[] = {137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 0};
        bad.write(reinterpret_cast<const char*>(bytes), 12);
    }
    throws([&] { read_image(output_dir / "corrupt.png"); }, "corrupt PNG");
}
void test_edges() {
    const auto image = read_image(assets / "synthetic/view0.png");
    Options o;
    const auto structures = make_edge_detector()->detect(image, o);
    require(!structures.empty(), "edge chains detected");
    bool curved = false;
    for (const auto& s : structures) {
        require(s.points.size() >= 3, "sampled chain length");
        for (std::size_t i = 1; i + 1 < s.points.size(); ++i)
            curved = curved ||
                     std::abs(cross(s.points[i] - s.points[0], s.points.back() - s.points[0])) > 5;
    }
    require(curved, "detector retains non-collinear curved samples");
}
void test_lineae_contract() {
    Image red(2, 2);
    for (std::size_t p = 0; p < 4; ++p)
        red.pixels[3 * p] = 255;
    const auto tensor = prepare_lineae_input(red, {2, 2}, "A");
    near(tensor[0], (1 - .538) / .257, 1e-5, "LINEA red mean/std");
    near(tensor[4], -.494 / .263, 1e-5, "LINEA green CHW");
    const auto large = prepare_lineae_input(red, {2, 2}, "XL");
    near(large[0], (1 - .485) / .229, 1e-5, "ImageNet profile");
    throws([] { lineae_normalization("unknown"); }, "unknown model variant");
    // A huge class-1 logit must not alter class-0 sigmoid confidence.
    auto lines = decode_lineae_output({0, 100}, {.1f, .2f, .9f, .8f}, {100, 50}, .49, 10);
    require(lines.size() == 1, "sigmoid of class zero, not softmax");
    near(lines[0].points.front().x, 10, 1e-5, "endpoint scaling x");
    near(lines[0].points.front().y, 10, 1e-5, "non-square endpoint scaling y");
    near(lines[0].confidence, .5, 1e-12, "sigmoid score");
    require(decode_lineae_output({0, -100}, {.1f, .2f, .9f, .8f}, {100, 50}, .6, 10).empty(),
            "confidence filtering");
    throws([] { decode_lineae_output({0, 0}, {0, 0, 1}, {100, 50}, .3, 10); },
           "output size mismatch");
    throws(
        [] {
            decode_lineae_output({std::numeric_limits<float>::quiet_NaN(), 0}, {0, 0, 1, 1},
                                 {100, 50}, .3, 10);
        },
        "nonfinite logits");
}
void test_device_contract() {
    std::vector<std::string> warnings;
    require(make_renderer(Device::cpu, warnings)->device_name() == "cpu", "explicit CPU");
    if (!cuda_available()) {
        throws([&] { make_renderer(Device::cuda, warnings); }, "strict unavailable CUDA");
        require(make_renderer(Device::automatic, warnings)->device_name() == "cpu" &&
                    !warnings.empty(),
                "reported auto fallback");
    }
    if (!lineae_compiled()) {
        Options o;
        throws([&] { make_lineae_detector(o); }, "missing optional LINEAE support");
    }
}
void test_failures() {
    const auto blank = read_image(assets / "synthetic/blank.png");
    throws([] { Stitcher{}.stitch({Image(2, 2), Image(2, 2)}); }, "tiny feature images");
    throws([&] { Stitcher{}.stitch({blank}); }, "too few images");
    throws([&] { Stitcher{}.stitch({blank, blank}); }, "blank pair");
    const auto a = read_image(assets / "synthetic/view0.png"),
               b = read_image(assets / "synthetic/view2.png");
    // Only 40 pixels overlap: cropping these two to disjoint 150px windows gives no valid scene
    // connection.
    Image left(150, 200), right(150, 200);
    for (int y = 0; y < 200; ++y)
        for (int x = 0; x < 150; ++x)
            for (int c = 0; c < 3; ++c) {
                left.pixels[(y * 150 + x) * 3 + c] = a.pixels[(y * a.width + x) * 3 + c];
                right.pixels[(y * 150 + x) * 3 + c] = b.pixels[(y * b.width + x + 150) * 3 + c];
            }
    throws([&] { Stitcher{}.stitch({left, right}); }, "non-overlapping images");
}
int optional_cuda() {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA renderer/device\n";
        return 77;
    }
    auto image = read_image(assets / "synthetic/view0.png");
    auto mesh = Mesh::regular(image.size(), 32);
    for (auto& point : mesh.deformed) {
        point.x += .3;
        point.y += .2;
    }
    Options o;
    const auto plan = detail::maps_from_meshes({mesh, mesh}, o);
    auto second = image;
    for (auto& value : second.pixels)
        value = static_cast<std::uint8_t>((3 * static_cast<unsigned>(value)) / 4 + 17);
    std::vector<std::string> warnings;
    const auto cpu = make_renderer(Device::cpu, warnings)->render({image, second}, plan, o);
    const auto gpu = make_renderer(Device::cuda, warnings)->render({image, second}, plan, o);
    require(cpu.mask == gpu.mask, "CPU/CUDA mask parity");
    int max_difference = 0;
    for (std::size_t k = 0; k < cpu.image.pixels.size(); ++k)
        max_difference =
            std::max(max_difference, std::abs(int(cpu.image.pixels[k]) - int(gpu.image.pixels[k])));
    require(max_difference <= 1, "CPU/CUDA RGB tolerance <= 1");
    return 0;
}
int optional_lineae() {
    const char* model = std::getenv("STITCHKIT_TEST_LINEAE_MODEL");
    const char* variant = std::getenv("STITCHKIT_TEST_LINEAE_VARIANT");
    if (!lineae_compiled() || !model || !variant) {
        std::cout << "SKIP: ONNX Runtime/model/variant not configured\n";
        return 77;
    }
    Options o;
    o.structures = StructureMode::lineae;
    o.lineae_model = model;
    o.lineae_variant = variant;
    const auto detector = make_lineae_detector(o);
    const auto shapes = detector->detect(limit_size(read_image(assets / "real/lib1.jpg"), 600), o);
    require(!shapes.empty(), "LINEAE inference plus contour augmentation completes");
    return 0;
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: stitchkit_tests test_name assets output\n";
        return 2;
    }
    assets = argv[2];
    output_dir = argv[3];
    std::filesystem::create_directories(output_dir);
    try {
        const std::string name = argv[1];
        if (name == "cuda_parity")
            return optional_cuda();
        if (name == "lineae_model")
            return optional_lineae();
        const std::map<std::string, std::function<void()>> tests{
            {"convolution", test_convolution},
            {"types", test_types},
            {"mesh", test_mesh},
            {"solver", test_solver},
            {"registration", test_registration},
            {"rew", test_rew},
            {"geometric_energy", test_geometric_energy},
            {"raster", test_raster},
            {"io", test_io},
            {"edges", test_edges},
            {"lineae_contract", test_lineae_contract},
            {"device_contract", test_device_contract},
            {"failures", test_failures}};
        const auto found = tests.find(name);
        if (found == tests.end())
            throw Error("Unknown test: " + name);
        found->second();
        std::cout << "PASS " << name << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
