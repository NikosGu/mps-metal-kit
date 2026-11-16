#include "mps_metal_kit/ops.hpp"
#include "mps_metal_kit/mps_bridge.hpp"

namespace mps_kit::ops {

void ragged_waves_mm_fwd(
    torch::Tensor X,
    std::vector<torch::Tensor> W_waves,
    std::vector<torch::Tensor> act_waves,
    std::vector<torch::Tensor> offs_waves,
    std::vector<torch::Tensor> starts_waves,
    std::vector<int64_t> total_toks_waves,
    torch::Tensor Y)
{
    int64_t T = X.size(1);
    int64_t D_IN = X.size(2);
    int64_t D_OUT = Y.size(2);
    size_t n_waves = W_waves.size();
    if (n_waves == 0) return;
    TORCH_CHECK(D_IN <= 256 && D_OUT <= 256, "ragged_waves_mm supports D_IN, D_OUT <= 256");

    id<MTLComputePipelineState> pso = PipelineRegistry::instance().pipeline("ragged", "ragged_seg_mm_fwd");
    auto bX = UnifiedBuffer::from_tensor(X);
    auto bY = UnifiedBuffer::from_tensor(Y);

    std::vector<UnifiedBuffer> bW(n_waves), bAct(n_waves), bOff(n_waves), bSt(n_waves);
    for (size_t w = 0; w < n_waves; ++w) {
        bW[w] = UnifiedBuffer::from_tensor(W_waves[w]);
        bAct[w] = UnifiedBuffer::from_tensor(act_waves[w]);
        bOff[w] = UnifiedBuffer::from_tensor(offs_waves[w]);
        bSt[w] = UnifiedBuffer::from_tensor(starts_waves[w]);
    }

    encode_on_torch_stream([&](id<MTLComputeCommandEncoder> enc) {
        [enc setComputePipelineState:pso];
        bX.bind(enc, 0);
        bY.bind(enc, 5);
        long t_c = (long)T, din_c = (long)D_IN, dout_c = (long)D_OUT;
        [enc setBytes:&t_c length:sizeof(long) atIndex:6];
        [enc setBytes:&din_c length:sizeof(long) atIndex:8];
        [enc setBytes:&dout_c length:sizeof(long) atIndex:9];

        for (size_t w = 0; w < n_waves; ++w) {
            int64_t tot = total_toks_waves[w];
            long n_act = (long)act_waves[w].size(0);
            if (tot <= 0 || n_act <= 0) continue;
            bW[w].bind(enc, 1);
            bAct[w].bind(enc, 2);
            bOff[w].bind(enc, 3);
            bSt[w].bind(enc, 4);
            [enc setBytes:&n_act length:sizeof(long) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tot, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    });
}

void ragged_waves_mm_bwd(
    torch::Tensor GY,
    torch::Tensor X,
    std::vector<torch::Tensor> W_waves,
    std::vector<torch::Tensor> act_waves,
    std::vector<torch::Tensor> offs_waves,
    std::vector<torch::Tensor> starts_waves,
    std::vector<torch::Tensor> stops_waves,
    std::vector<int64_t> total_toks_waves,
    torch::Tensor GX,
    std::vector<torch::Tensor> GW_waves)
{
    int64_t T = X.size(1);
    int64_t D_IN = X.size(2);
    int64_t D_OUT = GY.size(2);
    size_t n_waves = W_waves.size();
    if (n_waves == 0) return;

    auto& reg = PipelineRegistry::instance();
    id<MTLComputePipelineState> pso_dx = reg.pipeline("ragged", "ragged_seg_mm_dx_bwd");
    id<MTLComputePipelineState> pso_dw = reg.pipeline("ragged", "ragged_seg_mm_dw_bwd");

    auto bGY = UnifiedBuffer::from_tensor(GY);
    auto bX = UnifiedBuffer::from_tensor(X);
    auto bGX = UnifiedBuffer::from_tensor(GX);

    std::vector<UnifiedBuffer> bW(n_waves), bAct(n_waves), bOff(n_waves), bSt(n_waves), bSp(n_waves), bGW(n_waves);
    for (size_t w = 0; w < n_waves; ++w) {
        bW[w] = UnifiedBuffer::from_tensor(W_waves[w]);
        bAct[w] = UnifiedBuffer::from_tensor(act_waves[w]);
        bOff[w] = UnifiedBuffer::from_tensor(offs_waves[w]);
        bSt[w] = UnifiedBuffer::from_tensor(starts_waves[w]);
        bSp[w] = UnifiedBuffer::from_tensor(stops_waves[w]);
        bGW[w] = UnifiedBuffer::from_tensor(GW_waves[w]);
    }

    encode_on_torch_stream([&](id<MTLComputeCommandEncoder> enc) {
        long t_c = (long)T, din_c = (long)D_IN, dout_c = (long)D_OUT;

        [enc setComputePipelineState:pso_dx];
        bGY.bind(enc, 0);
        bGX.bind(enc, 5);
        [enc setBytes:&t_c length:sizeof(long) atIndex:6];
        [enc setBytes:&din_c length:sizeof(long) atIndex:8];
        [enc setBytes:&dout_c length:sizeof(long) atIndex:9];
        for (size_t w = 0; w < n_waves; ++w) {
            int64_t tot = total_toks_waves[w];
            long n_act = (long)act_waves[w].size(0);
            if (tot <= 0 || n_act <= 0) continue;
            bW[w].bind(enc, 1);
            bAct[w].bind(enc, 2);
            bOff[w].bind(enc, 3);
            bSt[w].bind(enc, 4);
            [enc setBytes:&n_act length:sizeof(long) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)tot, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }

        [enc setComputePipelineState:pso_dw];
        bGY.bind(enc, 0);
        bX.bind(enc, 1);
        [enc setBytes:&t_c length:sizeof(long) atIndex:6];
        [enc setBytes:&din_c length:sizeof(long) atIndex:7];
        [enc setBytes:&dout_c length:sizeof(long) atIndex:8];
        for (size_t w = 0; w < n_waves; ++w) {
            long n_act = (long)act_waves[w].size(0);
            if (n_act <= 0) continue;
            bAct[w].bind(enc, 2);
            bSt[w].bind(enc, 3);
            bSp[w].bind(enc, 4);
            bGW[w].bind(enc, 5);
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_act, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    });
}

} // namespace mps_kit::ops
