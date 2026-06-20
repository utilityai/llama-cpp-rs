#include "wrapper_common.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <stdint.h>

#include "llama.cpp/common/common.h"
#include "llama.cpp/common/fit.h"
#include "llama.cpp/common/json-schema-to-grammar.h"
#include "llama.cpp/common/speculative.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = nlohmann::ordered_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in libcommon).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
extern "C" int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    return static_cast<int>(common_fit_params(
        path_model,
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        log_level));
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    common_memory_breakdown_print(ctx);
}

// ---------------------------------------------------------------------------
// MTP speculative decoding shim over common_speculative.
// ---------------------------------------------------------------------------

static common_speculative * as_spec(llama_rs_speculative * h) {
    return reinterpret_cast<common_speculative *>(h);
}

extern "C" llama_rs_speculative * llama_rs_speculative_init_mtp(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float   p_min,
    bool    backend_sampling) {
    if (!ctx_tgt || !ctx_dft) {
        return nullptr;
    }
    try {
        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.draft.ctx_tgt          = ctx_tgt;
        params.draft.ctx_dft          = ctx_dft;
        params.draft.n_max            = n_max;
        params.draft.n_min            = n_min;
        params.draft.p_min            = p_min;
        params.draft.backend_sampling = backend_sampling;
        common_speculative * spec = common_speculative_init(params, /*n_seq=*/ 1);
        return reinterpret_cast<llama_rs_speculative *>(spec);
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_speculative_free(llama_rs_speculative * spec) {
    if (spec) {
        common_speculative_free(as_spec(spec));
    }
}

extern "C" int32_t llama_rs_speculative_n_max(const llama_rs_speculative * spec) {
    // common_speculative does not expose its configured n_max on the handle; the Rust
    // wrapper stores n_max from init and uses that to size the draft buffer. Kept for
    // ABI symmetry; returns -1 (unknown).
    (void) spec;
    return -1;
}

extern "C" bool llama_rs_speculative_need_embd_nextn(const llama_rs_speculative * spec) {
    if (!spec) {
        return false;
    }
    return common_speculative_need_embd_nextn(as_spec(const_cast<llama_rs_speculative *>(spec)));
}

extern "C" llama_rs_status llama_rs_speculative_begin(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    const llama_token * prompt, size_t prompt_len) {
    if (!spec) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_tokens p;
        if (prompt && prompt_len) {
            p.assign(prompt, prompt + prompt_len);
        }
        common_speculative_begin(as_spec(spec), seq_id, p);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_process(
    llama_rs_speculative * spec, const struct llama_batch * batch) {
    if (!spec || !batch) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        const bool ok = common_speculative_process(as_spec(spec), *batch);
        return ok ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_draft(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    llama_pos n_past, llama_token id_last,
    const llama_token * prompt, size_t prompt_len,
    llama_token * out_buf, size_t out_cap, size_t * out_len) {
    if (!spec || !out_buf || !out_len) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_tokens prompt_vec;
        if (prompt && prompt_len) {
            prompt_vec.assign(prompt, prompt + prompt_len);
        }
        llama_tokens result_vec;

        common_speculative_draft_params & dp = common_speculative_get_draft_params(as_spec(spec), seq_id);
        dp.drafting = true;
        dp.n_max    = -1;
        dp.n_past   = n_past;
        dp.id_last  = id_last;
        dp.prompt   = &prompt_vec;
        dp.result   = &result_vec;

        common_speculative_draft(as_spec(spec));

        *out_len = result_vec.size();
        if (result_vec.size() > out_cap) {
            return LLAMA_RS_STATUS_BUFFER_TOO_SMALL;
        }
        if (!result_vec.empty()) {
            std::memcpy(out_buf, result_vec.data(), result_vec.size() * sizeof(llama_token));
        }
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_speculative_accept(
    llama_rs_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    if (!spec) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        common_speculative_accept(as_spec(spec), seq_id, n_accepted);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
