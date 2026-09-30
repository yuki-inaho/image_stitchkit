#pragma once
#include "stitchkit/types.hpp"

namespace stitchkit {
struct LineaeNormalization {
    std::array<float, 3> mean;
    std::array<float, 3> standard_deviation;
};
// A/F/P/N/T use the LINEA training statistics; the other listed variants use ImageNet.
LineaeNormalization lineae_normalization(const std::string& variant);
// RGB uint8 -> resized, normalized planar CHW float32. No letterboxing.
std::vector<float> prepare_lineae_input(const Image& image, Size model_size,
                                        const std::string& variant);
// logits is [K,2]; normalized_endpoints is [K,4] in x1,y1,x2,y2 order.
// Confidence is sigmoid(logits[k,0]), NOT a two-class softmax.
std::vector<Structure> decode_lineae_output(const std::vector<float>& logits,
                                            const std::vector<float>& normalized_endpoints,
                                            Size original_size, double threshold, int max_lines,
                                            double minimum_length = 16.0);
} // namespace stitchkit
