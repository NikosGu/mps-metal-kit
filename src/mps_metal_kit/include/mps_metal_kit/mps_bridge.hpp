#pragma once
// Zero-Copy Unified-Memory Tensor Wrapper & Single-Encoder PyTorch MPSStream Helpers.

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

namespace mps_kit {

// Zero-copy view over a contiguous PyTorch `mps` tensor backed by Unified Memory (`MTLResourceStorageModeShared`).
// Exposes both the raw `id<MTLBuffer>` + byte offset for Metal compute encoders and the coherent host pointer
// `[buffer contents] + offset` for zero-copy CPU / Apple AMX reads and writes.
struct UnifiedBuffer {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    void* cpu_ptr = nullptr;

    static inline UnifiedBuffer from_tensor(const torch::Tensor& t) {
        TORCH_CHECK(t.device().is_mps(), "Expected tensor on 'mps' device, got ", t.device());
        TORCH_CHECK(t.is_contiguous(), "Expected contiguous MPS tensor");
        NSUInteger off = static_cast<NSUInteger>(t.storage_offset()) * static_cast<NSUInteger>(t.element_size());
        id<MTLBuffer> buf = __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
        TORCH_CHECK(buf != nil, "MPS tensor storage has null MTLBuffer");
        void* base = [buf contents];
        void* host = (base != nullptr) ? static_cast<void*>(static_cast<char*>(base) + off) : nullptr;
        return {buf, off, host};
    }

    template <typename T>
    inline T* data_as() const {
        TORCH_CHECK(cpu_ptr != nullptr, "MTLBuffer is not host-accessible (Shared/Unified storage required)");
        return reinterpret_cast<T*>(cpu_ptr);
    }

    inline void bind(id<MTLComputeCommandEncoder> enc, NSUInteger index, NSUInteger extra_offset = 0) const {
        [enc setBuffer:buffer offset:(offset + extra_offset) atIndex:index];
    }
};

// Encodes an arbitrary sequence of Metal compute dispatches onto PyTorch's active `MPSStream`
// inside ONE `id<MTLComputeCommandEncoder>`, preventing command-buffer splits and host syncs.
template <typename Fn>
inline void encode_on_torch_stream(Fn&& fn) {
    at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        @autoreleasepool {
            id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
            fn(enc);
        }
    });
}

// Commits and waits on PyTorch's active `MPSStream` so CPU / AMX routines can safely
// read and write `[MTLBuffer contents]` in place without copying tensors.
inline void sync_torch_stream() {
    at::mps::getCurrentMPSStream()->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
}

} // namespace mps_kit

#include "mps_metal_kit/shader_registry.hpp"
#include "mps_metal_kit/amx_lapack.hpp"
