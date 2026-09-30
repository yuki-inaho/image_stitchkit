#include <stitchkit/stitcher.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: stitchkit_example left.png right.png output.png\n";
        return 2;
    }
    try {
        stitchkit::Options options;
        options.method = stitchkit::Method::gesgsp;
        options.device = stitchkit::Device::cpu;
        const stitchkit::Stitcher pipeline(options);
        const auto result =
            pipeline.stitch({stitchkit::limit_size(stitchkit::read_image(argv[1]), 1000),
                             stitchkit::limit_size(stitchkit::read_image(argv[2]), 1000)});
        stitchkit::write_png(argv[3], result.image, result.mask);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
