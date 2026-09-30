#include <stitchkit/stitcher.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4)
        return 2;
    try {
        stitchkit::Options options;
        options.method = stitchkit::Method::rew;
        const auto result = stitchkit::Stitcher(options).stitch(
            {stitchkit::read_image(argv[1]), stitchkit::read_image(argv[2])});
        stitchkit::write_png(argv[3], result.image, result.mask);
        std::cout << stitchkit::version() << ": installed package consumer completed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
