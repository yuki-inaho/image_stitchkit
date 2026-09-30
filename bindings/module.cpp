// Python ownership and GIL operations belong in this adapter, never in the C++ core.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <stitchkit/stitcher.hpp>
#include <stitchkit/lineae.hpp>
#include <algorithm>
#include <cstring>
#include <memory>
#include <limits>

namespace nb = nanobind;
namespace sk = stitchkit;
using namespace nb::literals;
namespace {
std::string nanobind_version() {
    return std::to_string(NB_VERSION_MAJOR) + "." + std::to_string(NB_VERSION_MINOR) + "." +
           std::to_string(NB_VERSION_PATCH);
}
using InputArray = nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
using RgbArray = nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<3>, nb::c_contig>;
using MaskArray = nb::ndarray<nb::numpy, const std::uint8_t, nb::ndim<2>, nb::c_contig>;

sk::Image snapshot(const InputArray& a) {
    if (a.shape(2) != 3 || a.shape(0) > 32768 || a.shape(1) > 32768)
        throw nb::value_error("Expected uint8 RGB array with shape (H,W,3), dimensions 2..32768");
    sk::Image image(static_cast<int>(a.shape(1)), static_cast<int>(a.shape(0)));
    std::memcpy(image.pixels.data(), a.data(), image.pixels.size());
    return image;
}
std::vector<sk::Image> snapshots(const nb::list& arrays) {
    if (arrays.size() < 2 || arrays.size() > 16)
        throw nb::value_error("Expected 2 to 16 ordered RGB arrays");
    std::vector<sk::Image> images;
    images.reserve(arrays.size());
    for (const auto item : arrays) {
        // convert=false: do not silently truncate floats or reinterpret RGBA/BGR.
        const auto a = nb::cast<InputArray>(item, false);
        images.push_back(snapshot(a));
    }
    return images;
}
struct PythonResult {
    std::shared_ptr<const sk::Result> value;
    explicit PythonResult(sk::Result r) : value(std::make_shared<sk::Result>(std::move(r))) {}
};
template <typename T> nb::capsule owner(const std::shared_ptr<const T>& ptr) {
    auto held = std::make_unique<std::shared_ptr<const T>>(ptr);
    nb::capsule capsule(held.get(), [](void* p) noexcept {
        delete static_cast<std::shared_ptr<const T>*>(p);
    });
    held.release();
    return capsule;
}
RgbArray result_image(const PythonResult& r) {
    const auto& i = r.value->image;
    return RgbArray(i.pixels.data(), {std::size_t(i.height), std::size_t(i.width), 3}, owner(r.value));
}
MaskArray result_mask(const PythonResult& r) {
    const auto& i = r.value->image;
    return MaskArray(r.value->mask.data(), {std::size_t(i.height), std::size_t(i.width)}, owner(r.value));
}
RgbArray owned_image(sk::Image image) {
    std::shared_ptr<const sk::Image> held = std::make_shared<sk::Image>(std::move(image));
    return RgbArray(held->pixels.data(), {std::size_t(held->height), std::size_t(held->width), 3}, owner(held));
}
} // namespace
NB_MODULE(_native, m) {
    m.doc() = "Native stitching; input snapshots, immutable output views, explicit CPU/CUDA selection.";
    m.attr("__version__") = sk::version();
    m.attr("nanobind_version") = nanobind_version();
    nb::exception<sk::Error>(m, "StitchError", PyExc_RuntimeError);
    nb::enum_<sk::Method>(m, "Method").value("rew", sk::Method::rew)
        .value("niswgsp", sk::Method::niswgsp).value("gesgsp", sk::Method::gesgsp);
    nb::enum_<sk::Device>(m, "Device").value("cpu", sk::Device::cpu)
        .value("cuda", sk::Device::cuda).value("auto", sk::Device::automatic);
    nb::enum_<sk::StructureMode>(m, "StructureMode").value("edges", sk::StructureMode::edges)
        .value("lineae", sk::StructureMode::lineae).value("none", sk::StructureMode::none);
    auto options = nb::class_<sk::Options>(m, "Options").def(nb::init<>()).def("validate", &sk::Options::validate);
    options.def_rw("method", &sk::Options::method);
    options.def_rw("device", &sk::Options::device);
    options.def_rw("structures", &sk::Options::structures);
    options.def_rw("grid_size", &sk::Options::grid_size);
    options.def_rw("max_features", &sk::Options::max_features);
    options.def_rw("ratio_threshold", &sk::Options::ratio_threshold);
    options.def_rw("ransac_threshold", &sk::Options::ransac_threshold);
    options.def_rw("ransac_iterations", &sk::Options::ransac_iterations);
    options.def_rw("minimum_matches", &sk::Options::minimum_matches);
    options.def_rw("seed", &sk::Options::seed);
    options.def_rw("use_apap", &sk::Options::use_apap);
    options.def_rw("apap_sigma", &sk::Options::apap_sigma);
    options.def_rw("alignment_weight", &sk::Options::alignment_weight);
    options.def_rw("local_similarity_weight", &sk::Options::local_similarity_weight);
    options.def_rw("global_beta", &sk::Options::global_beta);
    options.def_rw("global_gamma", &sk::Options::global_gamma);
    options.def_rw("structure_weight", &sk::Options::structure_weight);
    options.def_rw("tps_lambda", &sk::Options::tps_lambda);
    options.def_rw("max_tps_controls", &sk::Options::max_tps_controls);
    options.def_rw("max_structures", &sk::Options::max_structures);
    options.def_rw("feather_width", &sk::Options::feather_width);
    options.def_rw("max_canvas_pixels", &sk::Options::max_canvas_pixels);
    options.def_rw("lineae_model", &sk::Options::lineae_model);
    options.def_rw("lineae_variant", &sk::Options::lineae_variant);
    options.def_rw("lineae_device", &sk::Options::lineae_device);
    options.def_rw("lineae_threshold", &sk::Options::lineae_threshold);
    nb::class_<sk::PreparedScene>(m, "PreparedScene")
        .def_prop_ro("image_count", &sk::PreparedScene::image_count)
        .def_prop_ro("registration_ms", &sk::PreparedScene::registration_ms)
        .def_prop_ro("feature_extractions", &sk::PreparedScene::feature_extractions);
    nb::class_<PythonResult>(m, "Result")
        .def_prop_ro("image", &result_image, nb::rv_policy::reference)
        .def_prop_ro("mask", &result_mask, nb::rv_policy::reference)
        .def_prop_ro("origin", [](const PythonResult& r) {
            return std::make_tuple(r.value->canvas.origin.x, r.value->canvas.origin.y);
        })
        .def("report_json", [](const PythonResult& r) { return sk::report_json(*r.value); })
        .def("save", [](const PythonResult& r, const std::filesystem::path& path) {
            nb::gil_scoped_release release;
            sk::write_png(path, r.value->image, r.value->mask);
        }, "path"_a);
    m.def("stitch", [](const nb::list& arrays, sk::Options options) {
        auto images = snapshots(arrays); // GIL held until the owned snapshot is complete.
        nb::gil_scoped_release release;
        return PythonResult(sk::Stitcher(std::move(options)).stitch(std::move(images)));
    }, "images"_a, "options"_a);
    m.def("prepare", [](const nb::list& arrays, sk::Options options) {
        auto images = snapshots(arrays);
        nb::gil_scoped_release release;
        return sk::Stitcher(std::move(options)).prepare(std::move(images));
    }, "images"_a, "options"_a);
    m.def("compose", [](const sk::PreparedScene& scene, sk::Options options) {
        nb::gil_scoped_release release;
        return PythonResult(sk::Stitcher(std::move(options)).compose(scene));
    }, "scene"_a, "options"_a);
    m.def("load", [](const std::filesystem::path& path, int max_side) {
        sk::Image image;
        {
            nb::gil_scoped_release release;
            image = sk::read_image(path);
            if (max_side < 0 || max_side == 1)
                throw sk::Error("max_side must be 0 (original size) or at least 2");
            if (max_side > 0) image = sk::limit_size(image, max_side);
        }
        return owned_image(std::move(image)); // Python array/capsule creation requires the GIL.
    }, "path"_a, "max_side"_a = 0);
    m.def("capabilities", [] {
        nb::dict d;
        d["version"] = sk::version();
        d["cxx_compiler"] = STITCHKIT_COMPILER_INFO;
        d["configuration"] = STITCHKIT_BUILD_CONFIGURATION;
        d["nanobind"] = nanobind_version();
        d["cuda_compiled"] = sk::cuda_compiled();
        d["cuda_available"] = sk::cuda_available();
        d["lineae_compiled"] = sk::lineae_compiled();
        d["gpu_scope"] = "sampling and compositing only";
        return d;
    });
}
