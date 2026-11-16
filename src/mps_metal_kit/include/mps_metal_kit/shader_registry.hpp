#pragma once
// Thread-Safe Runtime Metal Shading Language (MSL) Compiler & Compute Pipeline Cache.

#include <torch/extension.h>
#include <ATen/mps/MPSDevice.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace mps_kit {

class PipelineRegistry {
public:
    static PipelineRegistry& instance() {
        static PipelineRegistry reg;
        return reg;
    }

    void register_library(const std::string& lib_key, const std::string& msl_source, bool fast_math = true) {
        std::lock_guard<std::mutex> lock(mu_);
        if (libs_.find(lib_key) != libs_.end()) {
            return;
        }
        @autoreleasepool {
            id<MTLDevice> dev = at::mps::MPSDevice::getInstance()->device();
            NSError* err = nil;
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            opts.fastMathEnabled = fast_math ? YES : NO;
            NSString* src = [NSString stringWithUTF8String:msl_source.c_str()];
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
            TORCH_CHECK(
                lib != nil,
                "Metal shader compilation failed for '", lib_key, "': ",
                err ? [[err localizedDescription] UTF8String] : "unknown error"
            );
            libs_[lib_key] = lib;
        }
    }

    id<MTLComputePipelineState> pipeline(const std::string& lib_key, const std::string& fn_name) {
        std::string full_key = lib_key + "::" + fn_name;
        std::lock_guard<std::mutex> lock(mu_);
        auto it = psos_.find(full_key);
        if (it != psos_.end()) {
            return it->second;
        }
        auto lib_it = libs_.find(lib_key);
        TORCH_CHECK(lib_it != libs_.end(), "Metal library not registered: ", lib_key);
        @autoreleasepool {
            id<MTLDevice> dev = at::mps::MPSDevice::getInstance()->device();
            NSString* ns_fn = [NSString stringWithUTF8String:fn_name.c_str()];
            id<MTLFunction> fn = [lib_it->second newFunctionWithName:ns_fn];
            TORCH_CHECK(fn != nil, "Metal kernel function not found: ", full_key);
            NSError* err = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
            TORCH_CHECK(
                pso != nil,
                "Failed to build MTLComputePipelineState for '", full_key, "': ",
                err ? [[err localizedDescription] UTF8String] : "unknown error"
            );
            psos_[full_key] = pso;
            return pso;
        }
    }

private:
    std::mutex mu_;
    std::unordered_map<std::string, id<MTLLibrary>> libs_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> psos_;
};

} // namespace mps_kit
