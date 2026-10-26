#ifndef __SD_CORE_TENSOR_PARALLEL_H__
#define __SD_CORE_TENSOR_PARALLEL_H__

// Tensor parallelism (--split-mode tensor) on top of the GGML meta backend.
//
// The meta backend asks a callback how every statically allocated tensor is
// distributed across the simple devices. sd.cpp answers from a process-wide
// registry keyed by tensor pointer; tensors that were never registered are
// mirrored (a full copy on every device), which is always correct.

#include <map>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace sd::tp {

    // True when the backend / buffer is a GGML meta (tensor-parallel) object.
    bool backend_is_meta(ggml_backend_t backend);
    bool buffer_is_meta(ggml_backend_buffer_t buffer);
    size_t meta_device_count(ggml_backend_t backend);

    // Split-state callback handed to ggml_backend_meta_device().
    ggml_backend_meta_split_state get_split_state(const ggml_tensor* tensor, void* userdata);

    ggml_backend_meta_split_state mirrored_state();

    // Even split of `segments` (size, repeat count) along `axis`, each per-device
    // boundary rounded down to `granularity`; the last device takes the remainder.
    ggml_backend_meta_split_state split_state(int axis,
                                              const std::vector<std::pair<int64_t, uint32_t>>& segments,
                                              int64_t granularity,
                                              size_t n_devices);

    void register_tensor(const ggml_tensor* tensor, const ggml_backend_meta_split_state& state);
    void unregister_tensor(const ggml_tensor* tensor);
    // Drops the split states and base-name entries of every tensor in `ctx`.
    void unregister_context(const ggml_context* ctx);
    bool lookup_tensor(const ggml_tensor* tensor, ggml_backend_meta_split_state* state);

    // Base (model) weight split states by full tensor name, used to derive LoRA splits.
    void register_base_name(const std::string& name, const ggml_tensor* tensor);
    void unregister_base_name(const std::string& name);
    bool lookup_base_name(const std::string& name, const ggml_tensor** tensor, ggml_backend_meta_split_state* state);

    std::string describe(const ggml_backend_meta_split_state& state, size_t n_devices);

    // Assigns split states to the Qwen Image 2.1 DiT weights:
    //   attention q/k/v: output columns (whole heads), attention out: input rows,
    //   MLP gate/up: output columns (fused gate_up as two segments), MLP out: input rows,
    //   everything else (embeddings, modulation, norms, biases): mirrored.
    // Returns false (with `error`) if a tensor cannot be split evenly.
    bool assign_qwen_image_2_1_splits(const std::map<std::string, ggml_tensor*>& tensors,
                                      const std::string& prefix,
                                      int64_t head_dim,
                                      size_t n_devices,
                                      size_t* split_count,
                                      std::string* error);

    // Assigns split states to runtime LoRA tensors (names "lora.<base>[.N].<kind>")
    // so that their contribution has the same distribution as the base matmul.
    // Unsupported layouts on split base weights fail with `error`.
    bool assign_lora_splits(const std::map<std::string, ggml_tensor*>& lora_tensors,
                            size_t n_devices,
                            std::string* error);

    // Bytes every simple device needs for the tensors of `ctx` under their registered
    // split states (unregistered tensors are mirrored).
    std::vector<size_t> estimate_ctx_device_bytes(ggml_context* ctx, ggml_backend_t meta_backend);

    // Allocates every tensor of `ctx` on the meta backend's devices as WEIGHTS.
    // All tensors of `ctx` must be unallocated. Checks free device memory first because
    // the GGML meta allocator aborts on out-of-memory. Returns nullptr with `error` set.
    ggml_backend_buffer_t alloc_ctx_tensors(ggml_context* ctx,
                                            ggml_backend_t meta_backend,
                                            size_t headroom_bytes,
                                            size_t* total_bytes,
                                            std::string* error);
}  // namespace sd::tp

#endif  // __SD_CORE_TENSOR_PARALLEL_H__
