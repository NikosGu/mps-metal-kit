#include "mps_metal_kit/ops.hpp"
#include "mps_metal_kit/mps_bridge.hpp"

namespace mps_kit::ops {

void deltanet_chunk_scan_fwd(
    torch::Tensor M, torch::Tensor B_in, torch::Tensor DEC, torch::Tensor St0,
    torch::Tensor St_hist, torch::Tensor St_end,
    int64_t BH, int64_t N, int64_t DK, int64_t DV)
{
    if (BH <= 0 || N <= 0) return;
    TORCH_CHECK(DK >= 1 && DK <= 64 && DV >= 1 && DV <= 128, "deltanet_chunk_scan expects 1<=DK<=64, 1<=DV<=128");
    id<MTLComputePipelineState> pso = PipelineRegistry::instance().pipeline("chunk_scan", "deltanet_chunk_scan_fwd");
    auto bM = UnifiedBuffer::from_tensor(M);
    auto bB = UnifiedBuffer::from_tensor(B_in);
    auto bDEC = UnifiedBuffer::from_tensor(DEC);
    auto bS0 = UnifiedBuffer::from_tensor(St0);
    auto bSH = UnifiedBuffer::from_tensor(St_hist);
    auto bSE = UnifiedBuffer::from_tensor(St_end);

    encode_on_torch_stream([&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        bM.bind(enc, 0);
        bB.bind(enc, 1);
        bDEC.bind(enc, 2);
        bS0.bind(enc, 3);
        bSH.bind(enc, 4);
        bSE.bind(enc, 5);
        long n_c = (long)N, dk_c = (long)DK, dv_c = (long)DV;
        [enc setBytes:&n_c length:sizeof(long) atIndex:6];
        [enc setBytes:&dk_c length:sizeof(long) atIndex:7];
        [enc setBytes:&dv_c length:sizeof(long) atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)BH, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
}

void deltanet_chunk_scan_bwd(
    torch::Tensor M, torch::Tensor DEC, torch::Tensor St_hist,
    torch::Tensor G_hist, torch::Tensor G_end,
    torch::Tensor GM, torch::Tensor GB, torch::Tensor GDEC, torch::Tensor G_St0,
    int64_t BH, int64_t N, int64_t DK, int64_t DV)
{
    if (BH <= 0 || N <= 0) return;
    TORCH_CHECK(DK >= 1 && DK <= 64 && DV >= 1 && DV <= 128, "deltanet_chunk_scan expects 1<=DK<=64, 1<=DV<=128");
    id<MTLComputePipelineState> pso = PipelineRegistry::instance().pipeline("chunk_scan", "deltanet_chunk_scan_bwd");
    auto bM = UnifiedBuffer::from_tensor(M);
    auto bDEC = UnifiedBuffer::from_tensor(DEC);
    auto bSH = UnifiedBuffer::from_tensor(St_hist);
    auto bGH = UnifiedBuffer::from_tensor(G_hist);
    auto bGE = UnifiedBuffer::from_tensor(G_end);
    auto bGM = UnifiedBuffer::from_tensor(GM);
    auto bGB = UnifiedBuffer::from_tensor(GB);
    auto bGDEC = UnifiedBuffer::from_tensor(GDEC);
    auto bGS0 = UnifiedBuffer::from_tensor(G_St0);

    encode_on_torch_stream([&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        bM.bind(enc, 0);
        bDEC.bind(enc, 1);
        bSH.bind(enc, 2);
        bGH.bind(enc, 3);
        bGE.bind(enc, 4);
        bGM.bind(enc, 5);
        bGB.bind(enc, 6);
        bGDEC.bind(enc, 7);
        bGS0.bind(enc, 8);
        long n_c = (long)N, dk_c = (long)DK, dv_c = (long)DV;
        [enc setBytes:&n_c length:sizeof(long) atIndex:9];
        [enc setBytes:&dk_c length:sizeof(long) atIndex:10];
        [enc setBytes:&dv_c length:sizeof(long) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)BH, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
}

void token_delta_scan_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor BETA, torch::Tensor GAMMA, torch::Tensor ST0,
    torch::Tensor OUT, torch::Tensor CKPT, torch::Tensor ST_END,
    int64_t BH, int64_t T, int64_t DK, int64_t NCK)
{
    if (BH <= 0 || T <= 0) return;
    TORCH_CHECK(DK == 64 || DK == 128, "token_delta_scan supports DK in {64, 128}");
    std::string lib_key = (DK == 64) ? "token_scan_64" : "token_scan_128";
    NSUInteger tpb = (DK == 64) ? 128 : 256;
    id<MTLComputePipelineState> pso = PipelineRegistry::instance().pipeline(lib_key, "token_delta_scan_fwd");

    auto bQ = UnifiedBuffer::from_tensor(Q);
    auto bK = UnifiedBuffer::from_tensor(K);
    auto bV = UnifiedBuffer::from_tensor(V);
    auto bB = UnifiedBuffer::from_tensor(BETA);
    auto bG = UnifiedBuffer::from_tensor(GAMMA);
    auto bS0 = UnifiedBuffer::from_tensor(ST0);
    auto bO = UnifiedBuffer::from_tensor(OUT);
    auto bCK = UnifiedBuffer::from_tensor(CKPT);
    auto bSE = UnifiedBuffer::from_tensor(ST_END);

    encode_on_torch_stream([&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        bQ.bind(enc, 0);
        bK.bind(enc, 1);
        bV.bind(enc, 2);
        bB.bind(enc, 3);
        bG.bind(enc, 4);
        bS0.bind(enc, 5);
        bO.bind(enc, 6);
        bCK.bind(enc, 7);
        bSE.bind(enc, 8);
        long t_c = (long)T, nck_c = (long)NCK;
        [enc setBytes:&t_c length:sizeof(long) atIndex:9];
        [enc setBytes:&nck_c length:sizeof(long) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)BH, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpb, 1, 1)];
    });
}

void token_delta_scan_bwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor BETA, torch::Tensor GAMMA, torch::Tensor CKPT,
    torch::Tensor GOUT, torch::Tensor GST_END, torch::Tensor SCR,
    torch::Tensor GQ, torch::Tensor GK, torch::Tensor GV,
    torch::Tensor GBETA, torch::Tensor GGAMMA, torch::Tensor GST0,
    int64_t BH, int64_t T, int64_t DK, int64_t NCK)
{
    if (BH <= 0 || T <= 0) return;
    TORCH_CHECK(DK == 64 || DK == 128, "token_delta_scan supports DK in {64, 128}");
    std::string lib_key = (DK == 64) ? "token_scan_64" : "token_scan_128";
    NSUInteger tpb = (DK == 64) ? 128 : 256;
    id<MTLComputePipelineState> pso = PipelineRegistry::instance().pipeline(lib_key, "token_delta_scan_bwd");

    auto bQ = UnifiedBuffer::from_tensor(Q);
    auto bK = UnifiedBuffer::from_tensor(K);
    auto bV = UnifiedBuffer::from_tensor(V);
    auto bB = UnifiedBuffer::from_tensor(BETA);
    auto bG = UnifiedBuffer::from_tensor(GAMMA);
    auto bCK = UnifiedBuffer::from_tensor(CKPT);
    auto bGO = UnifiedBuffer::from_tensor(GOUT);
    auto bGSE = UnifiedBuffer::from_tensor(GST_END);
    auto bSCR = UnifiedBuffer::from_tensor(SCR);
    auto bGQ = UnifiedBuffer::from_tensor(GQ);
    auto bGK = UnifiedBuffer::from_tensor(GK);
    auto bGV = UnifiedBuffer::from_tensor(GV);
    auto bGB = UnifiedBuffer::from_tensor(GBETA);
    auto bGG = UnifiedBuffer::from_tensor(GGAMMA);
    auto bGS0 = UnifiedBuffer::from_tensor(GST0);

    encode_on_torch_stream([&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        bQ.bind(enc, 0);
        bK.bind(enc, 1);
        bV.bind(enc, 2);
        bB.bind(enc, 3);
        bG.bind(enc, 4);
        bCK.bind(enc, 5);
        bGO.bind(enc, 6);
        bGSE.bind(enc, 7);
        bSCR.bind(enc, 8);
        bGQ.bind(enc, 9);
        bGK.bind(enc, 10);
        bGV.bind(enc, 11);
        bGB.bind(enc, 12);
        bGG.bind(enc, 13);
        bGS0.bind(enc, 14);
        long t_c = (long)T, nck_c = (long)NCK;
        [enc setBytes:&t_c length:sizeof(long) atIndex:15];
        [enc setBytes:&nck_c length:sizeof(long) atIndex:16];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)BH, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpb, 1, 1)];
    });
}

} // namespace mps_kit::ops
