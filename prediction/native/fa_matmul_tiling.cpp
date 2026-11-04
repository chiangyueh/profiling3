#include <iostream>
#include "tiling/matmul/matmul_tiling.h"

int main()
{
    using namespace matmul_tiling;
    int m, n, k, om, on, ka, kb, fm, fn, fk, l1, l0c, ub, dtype, apos, cpos;
    int batches, seq1, seq2, n2, g, stage, batch_mode, same, unsplit;
    while (std::cin >> m >> n >> k >> om >> on >> ka >> kb >> fm >> fn >> fk >> l1 >> l0c >> ub
                    >> dtype >> apos >> cpos >> batches >> seq1 >> seq2 >> n2 >> g >> stage
                    >> batch_mode >> same >> unsplit) {
        PlatformInfo platform;
        platform.socVersion = platform_ascendc::SocVersion::ASCEND910B;
        platform.l1Size = l1;
        platform.l0CSize = l0c;
        platform.l0ASize = 64 * 1024;
        platform.l0BSize = 64 * 1024;
        platform.ubSize = ub;
        MatmulApiTiling mm(platform);
        auto dt = dtype == 2 ? DataType::DT_BF16 : DataType::DT_FLOAT16;
        mm.SetAType(apos ? TPosition::VECCALC : TPosition::GM, apos ? CubeFormat::NZ : CubeFormat::ND, dt, false);
        mm.SetBType(TPosition::GM, CubeFormat::ND, dt, stage == 1);
        mm.SetCType(TPosition::GM, CubeFormat::ND, cpos ? dt : DataType::DT_FLOAT);
        mm.SetShape(m, n, k);
        mm.SetOrgShape(om, on, ka, kb);
        if (batch_mode) {
            mm.SetALayout(batches, seq1, n2, g, stage == 1 ? k : seq2);
            mm.SetBLayout(batches, seq2, n2, 1, stage == 1 ? k : n);
            mm.SetCLayout(batches, seq1, n2, g, stage == 1 ? seq2 : n);
            mm.SetBatchNum(n2 * g);
        }
        mm.SetBias(false);
        mm.SetBufferSpace(l1, l0c);
        if (fm > 0 || fn > 0 || fk > 0) {
            mm.SetFixSplit(fm, fn, fk);
        }
        AscendC::tiling::TCubeTiling t{};
        if (mm.GetTiling(t) == -1) {
            std::cout << "ERROR" << std::endl;
            continue;
        }
        if (same && stage == 1 && t.baseK >= t.singleCoreK && t.depthA1 == 2) {
            t.depthA1 = 3;
        }
        if (unsplit && stage == 1) {
            t.baseK = ((k / 2 + 15) / 16) * 16;
            t.dbL0C = 2;
        }
        std::cout << t.baseM << ' ' << t.baseN << ' ' << t.baseK << ' ' << t.depthA1 << ' '
                  << t.depthB1 << ' ' << t.dbL0C << ' ' << t.stepM << ' ' << t.stepN << std::endl;
    }
}
