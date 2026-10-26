#include "core/tensor_parallel.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "core/util.h"
#include "ggml-backend-impl.h"

namespace sd::tp {

    namespace {
        struct Registry {
            std::mutex mutex;
            std::unordered_map<const ggml_tensor*, ggml_backend_meta_split_state> states;
            std::unordered_map<std::string, const ggml_tensor*> base_names;
        };

        Registry& registry() {
            static Registry instance;
            return instance;
        }

        // Every GGML block size divides 256, so 256-aligned boundaries never split a
        // quantization block and keep the CUDA kernels on their fast paths.
        constexpr int64_t kBlockGranularity = 256;

        bool ends_with_str(const std::string& value, const std::string& suffix) {
            return value.size() >= suffix.size() &&
                   value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        bool starts_with_str(const std::string& value, const std::string& prefix) {
            return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
        }

        bool every_device_nonempty(const ggml_backend_meta_split_state& state, size_t n_devices) {
            for (size_t j = 0; j < n_devices; ++j) {
                int64_t total = 0;
                for (uint32_t s = 0; s < state.n_segments; ++s) {
                    total += state.ne[s * n_devices + j] * state.nr[s];
                }
                if (total <= 0) {
                    return false;
                }
            }
            return true;
        }
    }  // namespace

    bool backend_is_meta(ggml_backend_t backend) {
        return backend != nullptr && ggml_backend_is_meta(backend);
    }

    bool buffer_is_meta(ggml_backend_buffer_t buffer) {
        return buffer != nullptr && ggml_backend_buffer_is_meta(buffer);
    }

    size_t meta_device_count(ggml_backend_t backend) {
        return backend_is_meta(backend) ? ggml_backend_meta_n_backends(backend) : 1;
    }

    ggml_backend_meta_split_state mirrored_state() {
        ggml_backend_meta_split_state state;
        memset(&state, 0, sizeof(state));
        state.axis       = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        state.nr[0]      = 1;
        state.n_segments = 1;
        return state;
    }

    ggml_backend_meta_split_state split_state(int axis,
                                              const std::vector<std::pair<int64_t, uint32_t>>& segments,
                                              int64_t granularity,
                                              size_t n_devices) {
        ggml_backend_meta_split_state state;
        memset(&state, 0, sizeof(state));
        GGML_ASSERT(axis >= 0 && axis < GGML_MAX_DIMS);
        GGML_ASSERT(!segments.empty() && segments.size() <= 16);
        GGML_ASSERT(n_devices > 0 && n_devices <= GGML_BACKEND_META_MAX_DEVICES);
        GGML_ASSERT(granularity > 0);
        state.axis = static_cast<ggml_backend_meta_split_axis>(axis);
        for (size_t s = 0; s < segments.size(); ++s) {
            const int64_t ne_s = segments[s].first;
            int64_t low        = 0;
            size_t j           = 0;
            for (; j + 1 < n_devices; ++j) {
                int64_t high = ne_s * static_cast<int64_t>(j + 1) / static_cast<int64_t>(n_devices);
                high -= high % granularity;
                high = std::max(high, low);
                state.ne[s * n_devices + j] = high - low;
                low                         = high;
            }
            state.ne[s * n_devices + j] = ne_s - low;
            state.nr[s]                 = segments[s].second;
        }
        state.n_segments = static_cast<uint32_t>(segments.size());
        return state;
    }

    ggml_backend_meta_split_state get_split_state(const ggml_tensor* tensor, void* userdata) {
        (void)userdata;
        ggml_backend_meta_split_state state;
        if (lookup_tensor(tensor, &state)) {
            return state;
        }
        return mirrored_state();
    }

    void register_tensor(const ggml_tensor* tensor, const ggml_backend_meta_split_state& state) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.states[tensor] = state;
    }

    void unregister_tensor(const ggml_tensor* tensor) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.states.erase(tensor);
    }

    void unregister_context(const ggml_context* ctx) {
        if (ctx == nullptr) {
            return;
        }
        std::unordered_set<const ggml_tensor*> tensors;
        auto* mutable_ctx = const_cast<ggml_context*>(ctx);
        for (ggml_tensor* t = ggml_get_first_tensor(mutable_ctx); t != nullptr; t = ggml_get_next_tensor(mutable_ctx, t)) {
            tensors.insert(t);
        }
        if (tensors.empty()) {
            return;
        }
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        for (const ggml_tensor* t : tensors) {
            reg.states.erase(t);
        }
        for (auto it = reg.base_names.begin(); it != reg.base_names.end();) {
            if (tensors.count(it->second) != 0) {
                it = reg.base_names.erase(it);
            } else {
                ++it;
            }
        }
    }

    bool lookup_tensor(const ggml_tensor* tensor, ggml_backend_meta_split_state* state) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto found = reg.states.find(tensor);
        if (found == reg.states.end()) {
            return false;
        }
        if (state != nullptr) {
            *state = found->second;
        }
        return true;
    }

    void register_base_name(const std::string& name, const ggml_tensor* tensor) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.base_names[name] = tensor;
    }

    void unregister_base_name(const std::string& name) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.base_names.erase(name);
    }

    bool lookup_base_name(const std::string& name, const ggml_tensor** tensor, ggml_backend_meta_split_state* state) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto found = reg.base_names.find(name);
        if (found == reg.base_names.end()) {
            return false;
        }
        if (tensor != nullptr) {
            *tensor = found->second;
        }
        if (state != nullptr) {
            auto split = reg.states.find(found->second);
            *state     = split == reg.states.end() ? mirrored_state() : split->second;
        }
        return true;
    }

    std::string describe(const ggml_backend_meta_split_state& state, size_t n_devices) {
        if (state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return "mirrored";
        }
        if (state.axis < 0 || state.axis >= GGML_MAX_DIMS) {
            return ggml_backend_meta_split_axis_name(state.axis);
        }
        std::string text = "axis" + std::to_string(static_cast<int>(state.axis)) + " {";
        for (uint32_t s = 0; s < state.n_segments; ++s) {
            if (s > 0) {
                text += "; ";
            }
            for (size_t j = 0; j < n_devices; ++j) {
                if (j > 0) {
                    text += ",";
                }
                text += std::to_string(state.ne[s * n_devices + j]);
            }
            if (state.nr[s] != 1) {
                text += " x" + std::to_string(state.nr[s]);
            }
        }
        return text + "}";
    }

    bool assign_qwen_image_2_1_splits(const std::map<std::string, ggml_tensor*>& tensors,
                                      const std::string& prefix,
                                      int64_t head_dim,
                                      size_t n_devices,
                                      size_t* split_count,
                                      std::string* error) {
        if (head_dim <= 0) {
            if (error != nullptr) {
                *error = "invalid head dimension";
            }
            return false;
        }
        const int64_t attn_granularity = std::lcm(head_dim, kBlockGranularity);
        const int64_t mlp_granularity  = kBlockGranularity;
        const std::string block_prefix = prefix + ".transformer_blocks.";
        size_t count                   = 0;

        for (const auto& [name, tensor] : tensors) {
            if (tensor == nullptr) {
                continue;
            }
            ggml_backend_meta_split_state state = mirrored_state();
            bool split                          = false;
            if (starts_with_str(name, block_prefix)) {
                auto rule = [&](const char* suffix) { return ends_with_str(name, suffix); };
                if (rule(".attn.to_q.weight") || rule(".attn.to_k.weight") || rule(".attn.to_v.weight")) {
                    state = split_state(GGML_BACKEND_SPLIT_AXIS_1, {{tensor->ne[1], 1}}, attn_granularity, n_devices);
                    split = true;
                } else if (rule(".attn.to_q.bias") || rule(".attn.to_k.bias") || rule(".attn.to_v.bias")) {
                    state = split_state(GGML_BACKEND_SPLIT_AXIS_0, {{tensor->ne[0], 1}}, attn_granularity, n_devices);
                    split = true;
                } else if (rule(".attn.to_out.0.weight")) {
                    state = split_state(GGML_BACKEND_SPLIT_AXIS_0, {{tensor->ne[0], 1}}, attn_granularity, n_devices);
                    split = true;
                } else if (rule(".img_mlp.gate_up.weight")) {
                    if (tensor->ne[1] % 2 != 0) {
                        if (error != nullptr) {
                            *error = "fused gate_up tensor '" + name + "' has an odd output size";
                        }
                        return false;
                    }
                    state = split_state(GGML_BACKEND_SPLIT_AXIS_1, {{tensor->ne[1] / 2, 2}}, mlp_granularity, n_devices);
                    split = true;
                } else if (rule(".img_mlp.proj.weight") || rule(".img_mlp.gate_layer.weight")) {
                    state = split_state(GGML_BACKEND_SPLIT_AXIS_1, {{tensor->ne[1], 1}}, mlp_granularity, n_devices);
                    split = true;
                } else if (rule(".img_mlp.out.weight")) {
                    state = split_state(GGML_BACKEND_SPLIT_AXIS_0, {{tensor->ne[0], 1}}, mlp_granularity, n_devices);
                    split = true;
                }
                // to_out.0.bias / img_mlp.out.bias (if present) are added after the
                // all-reduce and therefore stay mirrored.
            }
            if (split) {
                if (tensor->type == GGML_TYPE_I8) {
                    if (error != nullptr) {
                        *error = "tensor '" + name + "' uses INT8 tensorwise weights, which --split-mode tensor does not support";
                    }
                    return false;
                }
                if (ends_with_str(name, ".weight")) {
                    const std::string scale_name = name.substr(0, name.size() - 6) + "weight_scale";
                    auto scale                   = tensors.find(scale_name);
                    if (scale != tensors.end() && scale->second != nullptr && ggml_nelements(scale->second) > 1) {
                        if (error != nullptr) {
                            *error = "tensor '" + name + "' has a per-channel weight_scale, which --split-mode tensor does not support";
                        }
                        return false;
                    }
                }
                if (!every_device_nonempty(state, n_devices)) {
                    if (error != nullptr) {
                        *error = "tensor '" + name + "' is too small to split across " + std::to_string(n_devices) +
                                 " devices (" + describe(state, n_devices) + ")";
                    }
                    return false;
                }
                if (state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
                    const int64_t block = ggml_blck_size(tensor->type);
                    for (uint32_t s = 0; s < state.n_segments; ++s) {
                        for (size_t j = 0; j < n_devices; ++j) {
                            if (state.ne[s * n_devices + j] % block != 0) {
                                if (error != nullptr) {
                                    *error = "tensor '" + name + "' (" + ggml_type_name(tensor->type) +
                                             ") cannot be split on quantization block boundaries";
                                }
                                return false;
                            }
                        }
                    }
                }
                ++count;
            }
            register_tensor(tensor, state);
            register_base_name(name, tensor);
        }
        if (split_count != nullptr) {
            *split_count = count;
        }
        return true;
    }

    bool assign_lora_splits(const std::map<std::string, ggml_tensor*>& lora_tensors,
                            size_t n_devices,
                            std::string* error) {
        auto fail = [&](const std::string& message) {
            if (error != nullptr) {
                *error = message;
            }
            return false;
        };
        for (const auto& [name, tensor] : lora_tensors) {
            if (tensor == nullptr) {
                continue;
            }
            ggml_backend_meta_split_state state = mirrored_state();
            if (!starts_with_str(name, "lora.")) {
                register_tensor(tensor, state);
                continue;
            }
            const std::string body = name.substr(5);
            const size_t kind_pos  = body.rfind('.');
            if (kind_pos == std::string::npos) {
                register_tensor(tensor, state);
                continue;
            }
            const std::string kind = body.substr(kind_pos + 1);
            std::string base       = body.substr(0, kind_pos);
            int index              = 0;

            const ggml_tensor* base_tensor = nullptr;
            ggml_backend_meta_split_state base_state;
            bool found = lookup_base_name(base, &base_tensor, &base_state);
            if (!found) {
                const size_t index_pos = base.rfind('.');
                if (index_pos != std::string::npos && index_pos + 1 < base.size()) {
                    const std::string digits = base.substr(index_pos + 1);
                    bool numeric             = !digits.empty();
                    for (char c : digits) {
                        numeric = numeric && std::isdigit(static_cast<unsigned char>(c));
                    }
                    if (numeric && lookup_base_name(base.substr(0, index_pos), &base_tensor, &base_state)) {
                        found = true;
                        index = std::atoi(digits.c_str());
                        base  = base.substr(0, index_pos);
                    }
                }
            }
            if (!found) {
                // Qwen Image 2.1 LoRAs trained on the unfused MLP are remapped onto the
                // fused gate_up weight and concatenated, which a split weight cannot take.
                for (const char* unfused : {".img_mlp.gate_layer.weight", ".img_mlp.proj.weight"}) {
                    const size_t pos = base.find(unfused);
                    if (pos == std::string::npos) {
                        continue;
                    }
                    const std::string fused = base.substr(0, pos) + ".img_mlp.gate_up.weight";
                    ggml_backend_meta_split_state fused_state;
                    if (lookup_base_name(fused, nullptr, &fused_state) &&
                        fused_state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        return fail("LoRA tensor '" + name +
                                    "' targets the unfused MLP (gate_layer/proj) of a fused gate_up weight; "
                                    "this layout is not supported with --split-mode tensor");
                    }
                }
                register_tensor(tensor, state);
                continue;
            }
            if (base_state.axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                register_tensor(tensor, state);
                continue;
            }
            if (base_state.axis != GGML_BACKEND_SPLIT_AXIS_0 && base_state.axis != GGML_BACKEND_SPLIT_AXIS_1) {
                return fail("LoRA tensor '" + name + "' targets '" + base + "' with unsupported split " +
                            describe(base_state, n_devices));
            }
            if (index != 0) {
                return fail("LoRA tensor '" + name + "' adds a second (indexed) adapter to the split weight '" + base +
                            "'; indexed LoRA parts are not supported with --split-mode tensor");
            }
            const bool column_split = base_state.axis == GGML_BACKEND_SPLIT_AXIS_1;
            if (kind == "alpha" || kind == "scale" || ggml_nelements(tensor) == 1) {
                // Scalars only rescale the contribution.
                register_tensor(tensor, state);
                continue;
            }
            if (kind == "lora_mid") {
                register_tensor(tensor, state);
                continue;
            }
            if (kind == "lora_up") {
                if (!column_split) {
                    register_tensor(tensor, state);
                    continue;
                }
                if (ggml_n_dims(tensor) > 2 || tensor->ne[1] != base_tensor->ne[1]) {
                    return fail("LoRA tensor '" + name + "' does not match the output size of split weight '" + base + "'");
                }
                register_tensor(tensor, base_state);
                continue;
            }
            if (kind == "lora_down") {
                if (column_split) {
                    register_tensor(tensor, state);
                    continue;
                }
                if (ggml_n_dims(tensor) > 2 || tensor->ne[0] != base_tensor->ne[0]) {
                    return fail("LoRA tensor '" + name + "' does not match the input size of split weight '" + base + "'");
                }
                if (ggml_blck_size(tensor->type) > 1) {
                    for (uint32_t s = 0; s < base_state.n_segments; ++s) {
                        for (size_t j = 0; j < n_devices; ++j) {
                            if (base_state.ne[s * n_devices + j] % ggml_blck_size(tensor->type) != 0) {
                                return fail("LoRA tensor '" + name + "' cannot be split on quantization block boundaries");
                            }
                        }
                    }
                }
                register_tensor(tensor, base_state);
                continue;
            }
            return fail("LoRA tensor '" + name + "' (" + kind + ") patches the split weight '" + base +
                        "'; only lora_up/lora_down (+ lora_mid, alpha, scale) LoRAs are supported with --split-mode tensor");
        }
        return true;
    }

    std::vector<size_t> estimate_ctx_device_bytes(ggml_context* ctx, ggml_backend_t meta_backend) {
        const size_t n_devices = meta_device_count(meta_backend);
        std::vector<size_t> bytes(n_devices, 0);
        if (ctx == nullptr || !backend_is_meta(meta_backend)) {
            return bytes;
        }
        std::vector<size_t> alignment(n_devices, 256);
        for (size_t j = 0; j < n_devices; ++j) {
            ggml_backend_t simple = ggml_backend_meta_simple_backend(meta_backend, j);
            if (simple != nullptr) {
                alignment[j] = std::max<size_t>(1, ggml_backend_get_alignment(simple));
            }
        }
        for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            const size_t full = ggml_nbytes(t);
            ggml_backend_meta_split_state state = get_split_state(t, nullptr);
            for (size_t j = 0; j < n_devices; ++j) {
                size_t part = full;
                if (state.axis >= 0 && state.axis < GGML_MAX_DIMS && t->ne[state.axis] > 0) {
                    int64_t ne_j = 0;
                    for (uint32_t s = 0; s < state.n_segments; ++s) {
                        ne_j += state.ne[s * n_devices + j] * state.nr[s];
                    }
                    part = static_cast<size_t>((static_cast<double>(full) * static_cast<double>(ne_j)) /
                                               static_cast<double>(t->ne[state.axis])) + 1;
                }
                bytes[j] += GGML_PAD(part, alignment[j]);
            }
        }
        return bytes;
    }

    ggml_backend_buffer_t alloc_ctx_tensors(ggml_context* ctx,
                                            ggml_backend_t meta_backend,
                                            size_t headroom_bytes,
                                            size_t* total_bytes,
                                            std::string* error) {
        auto fail = [&](const std::string& message) -> ggml_backend_buffer_t {
            if (error != nullptr) {
                *error = message;
            }
            return nullptr;
        };
        if (ctx == nullptr || !backend_is_meta(meta_backend)) {
            return fail("tensor split allocation requires a meta backend");
        }
        size_t n_tensors = 0;
        for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data != nullptr || t->buffer != nullptr || t->view_src != nullptr) {
                return fail(std::string("tensor '") + t->name + "' is already allocated or a view; it cannot be tensor-split");
            }
            ++n_tensors;
        }
        if (n_tensors == 0) {
            if (total_bytes != nullptr) {
                *total_bytes = 0;
            }
            return fail("no tensors to allocate");
        }
        const std::vector<size_t> need = estimate_ctx_device_bytes(ctx, meta_backend);
        size_t total                   = 0;
        for (size_t j = 0; j < need.size(); ++j) {
            ggml_backend_t simple = ggml_backend_meta_simple_backend(meta_backend, j);
            ggml_backend_dev_t dev = simple != nullptr ? ggml_backend_get_device(simple) : nullptr;
            if (dev == nullptr) {
                return fail("tensor split device " + std::to_string(j) + " is unavailable");
            }
            size_t free_bytes  = 0;
            size_t total_bytes_dev = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes_dev);
            if (free_bytes < need[j] + headroom_bytes) {
                return fail("not enough free memory on " + std::string(ggml_backend_dev_name(dev)) + " for its tensor-split share: need " +
                            std::to_string(need[j] / (1024 * 1024)) + " MB + " + std::to_string(headroom_bytes / (1024 * 1024)) +
                            " MB headroom, free " + std::to_string(free_bytes / (1024 * 1024)) + " MB");
            }
            total += need[j];
        }
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(meta_backend);
        ggml_backend_buffer_t buffer    = ggml_backend_meta_alloc_ctx_tensors_from_buft(ctx, buft);
        if (buffer == nullptr) {
            return fail("meta buffer allocation failed");
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (total_bytes != nullptr) {
            *total_bytes = total;
        }
        return buffer;
    }

}  // namespace sd::tp
