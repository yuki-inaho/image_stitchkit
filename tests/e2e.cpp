#include <stitchkit/stitcher.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace stitchkit;
namespace {
void require(bool value, const std::string& message) {
    if (!value)
        throw Error("E2E: " + message);
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: stitchkit_e2e method case assets output\n";
        return 2;
    }
    try {
        Options o;
        o.method = parse_method(argv[1]);
        const std::string scenario = argv[2];
        const std::filesystem::path assets = argv[3], out = argv[4];
        std::filesystem::create_directories(out);
        std::vector<Image> images;
        Image truth;
        bool reference = false;
        if (scenario == "translation" || scenario == "chain" || scenario == "parallax") {
            images.push_back(read_image(assets / "synthetic/view0.png"));
            images.push_back(read_image(assets / (scenario == "parallax" ? "synthetic/parallax1.png"
                                                                         : "synthetic/view1.png")));
            if (scenario == "chain")
                images.push_back(read_image(assets / "synthetic/view2.png"));
            if (scenario != "parallax") {
                truth = read_image(assets / (scenario == "chain" ? "synthetic/scene.png"
                                                                 : "synthetic/pair_truth.png"));
                reference = true;
            }
        } else if (scenario == "library" || scenario == "town_hall" || scenario == "ship") {
            const std::string stem = scenario == "library" ? "lib" : scenario;
            images.push_back(limit_size(read_image(assets / ("real/" + stem + "1.jpg")), 640));
            images.push_back(limit_size(read_image(assets / ("real/" + stem + "2.jpg")), 640));
        } else
            throw Error("Unknown E2E scenario");
        const auto result = Stitcher(o).stitch(images);
        const auto& d = result.diagnostics;
        const std::string stem = scenario + "_" + to_string(o.method);
        write_png(out / (stem + ".png"), result.image, result.mask);
        write_report(out / (stem + ".json"), result);
        const auto roundtrip = read_image(out / (stem + ".png"));
        require(roundtrip.width == result.image.width, "encoded panorama dimensions");
        require(d.feature_matches >= 10 && std::isfinite(d.alignment_rmse_after),
                "verified correspondences and finite residual");
        for (std::size_t p = 0; p < result.mask.size(); ++p)
            if (result.mask[p])
                for (int c = 0; c < 3; ++c)
                    require(roundtrip.pixels[3 * p + c] == result.image.pixels[3 * p + c],
                            "valid PNG RGB survives alpha encoding");
        const auto valid = std::count(result.mask.begin(), result.mask.end(), 255);
        require(valid > static_cast<long>(images[0].pixel_count() * .9),
                "output retains substantial input coverage");
        require(result.image.width < 4 * (images[0].width + images[1].width) &&
                    result.image.height < 4 * (images[0].height + images[1].height),
                "bounded output geometry");
        if (o.method == Method::rew)
            require(d.tps_controls >= 6 && d.structure_equations == 0, "REW actually uses TPS");
        else
            require(d.equations > d.unknowns && d.dense_matches > 0,
                    "mesh solve and moving DLT actually execute");
        if (o.method == Method::gesgsp)
            require(d.structures > 0 && d.structure_equations > 0,
                    "GES uses structure constraints");
        if (o.method == Method::niswgsp)
            require(d.structure_equations == 0, "NIS does not execute GES constraints");
        if (scenario == "parallax")
            require(d.alignment_rmse_after < 3.0, "synthetic local parallax residual bound");
        if (reference) {
            double squared = 0;
            std::size_t components = 0, covered = 0;
            int black_valid = 0;
            const int ox = static_cast<int>(std::lround(result.canvas.origin.x)),
                      oy = static_cast<int>(std::lround(result.canvas.origin.y));
            for (int y = 3; y < truth.height - 3; ++y)
                for (int x = 3; x < truth.width - 3; ++x) {
                    const int px = x - ox, py = y - oy;
                    if (px < 0 || py < 0 || px >= result.image.width || py >= result.image.height)
                        continue;
                    const auto p = std::size_t(py) * result.image.width + px,
                               q = std::size_t(y) * truth.width + x;
                    if (!result.mask[p])
                        continue;
                    ++covered;
                    for (int c = 0; c < 3; ++c) {
                        const double delta =
                            int(result.image.pixels[3 * p + c]) - int(truth.pixels[3 * q + c]);
                        squared += delta * delta;
                        ++components;
                    }
                    if (x >= 8 && x <= 20 && y >= 110 && y <= 170 && result.mask[p] &&
                        result.image.pixels[3 * p] == 0)
                        ++black_valid;
                }
            const double rmse = std::sqrt(squared / std::max<std::size_t>(1, components));
            require(covered >
                        static_cast<std::size_t>((truth.width - 6) * (truth.height - 6) * .985),
                    "ground-truth canvas coverage >= 98.5%");
            require(rmse < 6.0, "ground-truth RGB RMSE < 6 intensity levels");
            require(black_valid > 500, "valid black patch survives PNG alpha masking");
            require(d.alignment_rmse_after < .6, "known-translation correspondence RMSE < 0.6 px");
            std::ofstream metrics(out / (stem + "_quality.json"));
            metrics << "{\"rgb_rmse\":" << rmse << ",\"covered_pixels\":" << covered
                    << ",\"black_valid\":" << black_valid << "}\n";
            std::cout << "RGB RMSE=" << rmse << ", coverage=" << covered << '\n';
        }
        std::cout << "PASS " << stem << " size=" << result.image.width << 'x' << result.image.height
                  << " matches=" << d.feature_matches
                  << " alignment_rmse=" << d.alignment_rmse_after << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
