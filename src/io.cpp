#include "stitchkit/stitcher.hpp"
#include "internal.hpp"
#include "jpeg_bridge.h"
#include <png.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>

namespace stitchkit {
namespace {
struct FileCloser {
    void operator()(FILE* file) const noexcept {
        if (file)
            std::fclose(file);
    }
};
using File = std::unique_ptr<FILE, FileCloser>;
File open_file(const std::filesystem::path& path, bool write) {
#ifdef _WIN32
    FILE* f = _wfopen(path.c_str(), write ? L"wb" : L"rb");
#else
    FILE* f = std::fopen(path.c_str(), write ? "wb" : "rb");
#endif
    if (!f)
        throw Error("Cannot open file: " + path.u8string());
    return File(f);
}
void parents(const std::filesystem::path& p) {
    if (!p.parent_path().empty())
        std::filesystem::create_directories(p.parent_path());
}
} // namespace
Image read_image(const std::filesystem::path& path) {
    auto file = open_file(path, false);
    std::array<unsigned char, 8> signature{};
    if (std::fread(signature.data(), 1, signature.size(), file.get()) != signature.size())
        throw Error("Image file is empty or truncated: " + path.u8string());
    std::rewind(file.get());
    if (png_sig_cmp(signature.data(), 0, 8) == 0) {
        png_image png{};
        png.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_stdio(&png, file.get())) {
            const std::string e = png.message;
            png_image_free(&png);
            throw Error("PNG header: " + e);
        }
        struct Guard {
            png_image* p;
            ~Guard() { png_image_free(p); }
        } guard{&png};
        Image im(static_cast<int>(png.width), static_cast<int>(png.height));
        png.format = PNG_FORMAT_RGB;
        const png_color background{255, 255, 255};
        if (!png_image_finish_read(&png, &background, im.pixels.data(), 0, nullptr))
            throw Error("PNG decoding: " + std::string(png.message));
        return im;
    }
    if (signature[0] == 0xff && signature[1] == 0xd8) {
        int w = 0, h = 0;
        char error[256] = {};
        std::unique_ptr<unsigned char, decltype(&std::free)> data(
            stitchkit_decode_jpeg(file.get(), &w, &h, error), &std::free);
        if (!data)
            throw Error("JPEG decoding: " + std::string(error));
        Image im(w, h);
        std::copy_n(data.get(), im.pixels.size(), im.pixels.data());
        return im;
    }
    throw Error("Unsupported image format (PNG and JPEG are supported): " + path.u8string());
}
void write_png(const std::filesystem::path& path, const Image& im,
               const std::vector<std::uint8_t>& mask) {
    im.validate();
    if (!mask.empty() && mask.size() != im.pixel_count())
        throw Error("Invalid output mask size");
    parents(path);
    auto file = open_file(path, true);
    png_image png{};
    png.version = PNG_IMAGE_VERSION;
    png.width = static_cast<png_uint_32>(im.width);
    png.height = static_cast<png_uint_32>(im.height);
    png.format = mask.empty() ? PNG_FORMAT_RGB : PNG_FORMAT_RGBA;
    std::vector<std::uint8_t> rgba;
    const void* data = im.pixels.data();
    if (!mask.empty()) {
        rgba.resize(im.pixel_count() * 4);
        for (std::size_t i = 0; i < mask.size(); ++i) {
            for (int c = 0; c < 3; ++c)
                rgba[i * 4 + c] = im.pixels[i * 3 + c];
            rgba[i * 4 + 3] = mask[i];
        }
        data = rgba.data();
    }
    if (!png_image_write_to_stdio(&png, file.get(), 0, data, 0, nullptr))
        throw Error("PNG encoding: " + std::string(png.message));
    if (std::fflush(file.get()) != 0)
        throw Error("Failed to flush output image");
}
Image resize_image(const Image& input, Size target) {
    input.validate();
    Image output(target.width, target.height);
    // Pixel-centre convention matches OpenCV's INTER_LINEAR coordinate transform.
    for (int y = 0; y < target.height; ++y)
        for (int x = 0; x < target.width; ++x) {
            const double sx = std::clamp((x + 0.5) * input.width / target.width - 0.5, 0.0,
                                         double(input.width - 1));
            const double sy = std::clamp((y + 0.5) * input.height / target.height - 0.5, 0.0,
                                         double(input.height - 1));
            const auto p = detail::sample(input, sx, sy);
            for (int c = 0; c < 3; ++c)
                output.pixels[(static_cast<std::size_t>(y) * target.width + x) * 3 + c] =
                    static_cast<std::uint8_t>(std::clamp(std::lround(p[c]), 0L, 255L));
        }
    return output;
}
Image limit_size(const Image& input, int max_side) {
    input.validate();
    if (max_side < 2)
        throw Error("max_side must be >=2");
    const int longest = std::max(input.width, input.height);
    if (longest <= max_side)
        return input;
    const double scale = double(max_side) / longest;
    return resize_image(input, {std::max(2, int(std::lround(input.width * scale))),
                                std::max(2, int(std::lround(input.height * scale)))});
}
std::string version() { return STITCHKIT_VERSION; }
} // namespace stitchkit
