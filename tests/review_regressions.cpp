#include <stitchkit/stitcher.hpp>
#include <cmath>
#include <climits>
#include <filesystem>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <string>
#include <utility>
using namespace stitchkit;
namespace {
void check(bool ok, const char* message) { if (!ok) throw Error(message); }
template <class F> void rejects(F fn) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    check(caught, "Expected exception was not raised");
}
void atomic_row() {
    LeastSquaresProblem p; p.columns = 2;
    p.add_row({{0, 1}}, 2);
    const auto count = p.coefficients.size(), rows = p.rhs.size();
    rejects([&] { p.add_row({{0, 99}, {2, 1}}, 0); });
    check(p.coefficients.size() == count && p.rhs.size() == rows, "Rejected row changed the matrix");
    p.add_row({{1, 1}}, 3);
    const auto x = make_eigen_solver()->solve(p);
    check(std::abs(x.values[0] - 2) < 1e-12 && std::abs(x.values[1] - 3) < 1e-12,
          "Recovery after rejected row is incorrect");
}
void weighted_overflow() {
    LeastSquaresProblem p; p.columns = 1;
    const double large = std::numeric_limits<double>::max();
    rejects([&] { p.add_row({{0, large}}, 1, 2); });
    rejects([&] { p.add_row({{0, 1}}, large, 2); });
    check(p.rhs.empty() && p.coefficients.empty(), "Overflow polluted the matrix");
    p.add_row({{0, 1}}, 3);
    check(std::abs(make_eigen_solver()->solve(p).values[0] - 3) < 1e-12, "Recovery failed");
}
void mesh_bounds() {
    const auto m = Mesh::regular({128, 96}, INT_MAX);
    check(m.columns == 2 && m.rows == 2, "Ceiling division overflow");
    rejects([] { Mesh::regular({INT_MAX, 2}, 1); });
    rejects([] { Mesh::regular({32768, 32768}, 1); });
    rejects([] { Mesh::regular({1, 2}, 4); });
}
void renderer_enum() {
    std::vector<std::string> warnings;
    rejects([&] { make_renderer(static_cast<Device>(1234), warnings); });
    check(warnings.empty(), "Invalid device must not silently fall back");
}
struct Comma : std::numpunct<char> { char do_decimal_point() const override { return ','; } };
struct RestoreLocale {
    std::locale previous = std::locale();
    ~RestoreLocale() { std::locale::global(previous); }
};
void report_locale() {
    Result r; r.image = Image(2, 2); r.mask.assign(4, 255); r.canvas.width=2; r.canvas.height=2;
    r.diagnostics.structure_ms=1.25; r.diagnostics.method="rew";
    RestoreLocale restore;
    std::locale::global(std::locale(std::locale::classic(), new Comma));
    const auto json = report_json(r);
    check(json.find("1.25") != std::string::npos, "JSON inherited a decimal-comma locale");
    check(json.find("\"schema_version\": 2") != std::string::npos, "Schema version missing");
}
std::vector<Image> input(const std::filesystem::path& assets, int count=2) {
    std::vector<Image> images;
    for (int i=0;i<count;++i) images.push_back(limit_size(read_image(assets/"synthetic"/("view"+std::to_string(i)+".png")),256));
    return images;
}
void prepared_scene(const std::filesystem::path& assets) {
    Options o; o.method=Method::rew; o.max_features=700;
    Stitcher engine(o);
    auto images=input(assets);
    const auto direct=engine.stitch(images);
    auto prepared=engine.prepare(images);
    check(prepared.image_count()==2 && prepared.feature_extractions()==2, "Preparation counters wrong");
    images[0].pixels.assign(images[0].pixels.size(),0);
    const auto reused=engine.compose(prepared);
    check(direct.image.pixels==reused.image.pixels && direct.mask==reused.mask, "Prepared ownership/parity violated");
    check(reused.diagnostics.registration_reused && reused.diagnostics.registration_ms==0 &&
          reused.diagnostics.feature_extractions==0 && reused.diagnostics.prepared_registration_ms>0,
          "Reused/historical timing not distinguished");
    auto changed=o; changed.seed++;
    rejects([&] { Stitcher(changed).compose(prepared); });
    changed=o; changed.method=Method::niswgsp;
    check(Stitcher(changed).compose(prepared).diagnostics.method=="niswgsp", "Mode switch failed");
    auto moved=std::move(prepared);
    rejects([&] { prepared.image_count(); });
    rejects([&] { engine.compose(prepared); });
    check(moved.image_count()==2,"Move lost valid snapshot");
}
void batch_features(const std::filesystem::path& assets) {
    const auto images=input(assets,3);
    auto matcher=make_sift_matcher(); Options o; o.max_features=700;
    auto batch=matcher->match_sequence(images,o);
    check(batch.feature_extractions==3 && batch.pairs.size()==2, "Feature extraction is not once per image");
    for (std::size_t i=1;i<images.size();++i) {
        const auto pair=matcher->match(images[i],images[i-1],o);
        check(pair.matches.size()==batch.pairs[i-1].matches.size(),"Batch matching changed correspondences");
        for(int j=0;j<9;++j) check(std::abs(pair.homography.values[j]-batch.pairs[i-1].homography.values[j])<1e-12,
                                  "Batch matching changed the transform");
    }
}
class FakeMatcher final : public IFeatureMatcher {
 public:
    PairRegistration match(const Image&,const Image&,const Options&) const override {
        PairRegistration pair;
        for (int y=16;y<=64;y+=16) for (int x=16;x<=80;x+=16)
            pair.matches.push_back({{double(x),double(y)},{double(x),double(y)},1});
        return pair;
    }
};
class CountingRenderer final : public IRenderer {
 public:
    mutable int calls=0;
    Result render(const std::vector<Image>& images,const WarpPlan& plan,const Options& o) const override {
        ++calls; std::vector<std::string> warnings;
        return make_renderer(Device::cpu,warnings)->render(images,plan,o);
    }
    std::string device_name() const override { return "test-cpu"; }
};
void injection() {
    Options o; o.method=Method::niswgsp; o.use_apap=false;
    auto renderer=std::make_shared<CountingRenderer>();
    Stitcher engine(o,std::make_unique<FakeMatcher>(),nullptr,make_eigen_solver(),renderer);
    const auto result=engine.stitch({Image(128,96,20),Image(128,96,20)});
    check(renderer->calls==1 && result.diagnostics.actual_device=="test-cpu","Injected renderer was bypassed");
    check(result.diagnostics.feature_extractions==-1,"Unknown extraction count was invented");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw Error("Usage: review_regressions test assets");
        const std::string name=argv[1];
        if(name=="atomic_row") atomic_row();
        else if(name=="weighted_overflow") weighted_overflow();
        else if(name=="mesh_bounds") mesh_bounds();
        else if(name=="renderer_enum") renderer_enum();
        else if(name=="report_locale") report_locale();
        else if(name=="prepared_scene") prepared_scene(argv[2]);
        else if(name=="batch_features") batch_features(argv[2]);
        else if(name=="injection") injection();
        else throw Error("Unknown test");
        std::cout<<"PASS "<<name<<'\n'; return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
