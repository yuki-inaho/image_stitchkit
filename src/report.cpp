#include "stitchkit/stitcher.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>

namespace stitchkit {
namespace {
std::string escape(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c == '\n')
            out += "\\n";
        else if (c == '\r')
            out += "\\r";
        else if (c == '\t')
            out += "\\t";
        else if (c < 32) {
            const char* hex = "0123456789abcdef";
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 15];
        } else
            out += static_cast<char>(c);
    }
    return out + '"';
}
} // namespace
std::string report_json(const Result& r) {
    std::ostringstream f;
    f.imbue(std::locale::classic());
    const auto& d = r.diagnostics;
    f << std::setprecision(12)
      << "{\n  \"schema_version\": 2,\n  \"version\": " << escape(version())
      << ",\n  \"method\": " << escape(d.method)
      << ",\n  \"registration_reused\": " << (d.registration_reused ? "true" : "false")
      << ",\n  \"feature_extractions\": " << d.feature_extractions
      << ",\n  \"prepared_registration_ms\": " << d.prepared_registration_ms
      << ",\n  \"requested_device\": " << escape(d.requested_device)
      << ",\n  \"actual_device\": " << escape(d.actual_device)
      << ",\n  \"line_device\": " << escape(d.line_device) << ",\n  \"width\": " << r.image.width
      << ",\n  \"height\": " << r.image.height << ",\n  \"origin\": [" << r.canvas.origin.x << ", "
      << r.canvas.origin.y << "]"
      << ",\n  \"valid_pixels\": " << std::count(r.mask.begin(), r.mask.end(), std::uint8_t(255))
      << ",\n  \"feature_matches\": " << d.feature_matches
      << ",\n  \"dense_matches\": " << d.dense_matches << ",\n  \"structures\": " << d.structures
      << ",\n  \"structure_equations\": " << d.structure_equations
      << ",\n  \"tps_controls\": " << d.tps_controls << ",\n  \"equations\": " << d.equations
      << ",\n  \"unknowns\": " << d.unknowns
      << ",\n  \"alignment_rmse_before\": " << d.alignment_rmse_before
      << ",\n  \"alignment_rmse_after\": " << d.alignment_rmse_after
      << ",\n  \"solver_relative_residual\": " << d.solver_relative_residual
      << ",\n  \"timing_ms\": {\"registration\": " << d.registration_ms
      << ", \"structure\": " << d.structure_ms << ", \"geometry\": " << d.geometry_ms << ", \"render\": " << d.render_ms << ", \"total\": " << d.total_ms
      << "},\n  \"warnings\": [";
    for (std::size_t i = 0; i < d.warnings.size(); ++i) {
        if (i)
            f << ", ";
        f << escape(d.warnings[i]);
    }
    f << "],\n  \"processed_input_sizes\": [";
    for (std::size_t i = 0; i < d.input_sizes.size(); ++i) {
        if (i)
            f << ", ";
        f << "[" << d.input_sizes[i].width << ", " << d.input_sizes[i].height << "]";
    }
    const auto& o = d.options;
    f << "],\n  \"options\": {"
      << "\"grid_size\":" << o.grid_size << ",\"max_features\":" << o.max_features
      << ",\"ratio_threshold\":" << o.ratio_threshold
      << ",\"ransac_threshold\":" << o.ransac_threshold
      << ",\"ransac_iterations\":" << o.ransac_iterations
      << ",\"minimum_matches\":" << o.minimum_matches << ",\"seed\":" << o.seed
      << ",\"use_apap\":" << (o.use_apap ? "true" : "false") << ",\"apap_sigma\":" << o.apap_sigma
      << ",\"alignment_weight\":" << o.alignment_weight
      << ",\"local_similarity_weight\":" << o.local_similarity_weight
      << ",\"global_beta\":" << o.global_beta << ",\"global_gamma\":" << o.global_gamma
      << ",\"structure_weight\":" << o.structure_weight << ",\"tps_lambda\":" << o.tps_lambda
      << ",\"max_tps_controls\":" << o.max_tps_controls
      << ",\"max_structures\":" << o.max_structures << ",\"feather_width\":" << o.feather_width
      << ",\"max_canvas_pixels\":" << o.max_canvas_pixels << ",\"structure_detector\":"
      << escape(o.structures == StructureMode::lineae  ? "lineae"
                : o.structures == StructureMode::edges ? "edges"
                                                       : "none")
      << ",\"lineae_variant\":" << escape(o.lineae_variant)
      << ",\"lineae_threshold\":" << o.lineae_threshold << "}\n}\n";

    return f.str();
}
void write_report(const std::filesystem::path& path, const Result& r) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path);
    if (!f)
        throw Error("Cannot write report: " + path.u8string());
    f << report_json(r);
    f.flush();
    if (!f)
        throw Error("Report write failed");
}
} // namespace stitchkit
