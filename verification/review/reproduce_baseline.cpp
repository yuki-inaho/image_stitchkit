// Compile against v0.1.0 to reproduce the rejected-row state corruption.
#include <stitchkit/stitcher.hpp>
#include <iostream>
int main() {
    stitchkit::LeastSquaresProblem p; p.columns=2; p.add_row({{0,1}},2);
    const auto before=p.coefficients.size();
    try { p.add_row({{0,99},{2,1}},0); } catch (const std::exception&) {}
    std::cout<<"coefficient_count_before="<<before<<" after_rejected_row="<<p.coefficients.size()<<'\n';
    return p.coefficients.size()==before?0:1;
}
