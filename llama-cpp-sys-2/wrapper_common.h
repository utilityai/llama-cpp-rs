#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct llama_model;
struct llama_sampler;
struct llama_vocab;

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

// Fit model/context params to device memory (wraps llama.cpp's common_fit_params).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);

void llama_rs_memory_breakdown_print(const struct llama_context * ctx);

void llama_rs_string_free(char * ptr);

// MTP (NextN) speculative decoding shim over llama.cpp's common_speculative (draft-mtp).
// The handle is internally a common_speculative *.
typedef struct llama_rs_speculative llama_rs_speculative;

// Returns NULL on failure (null contexts or C++ exception).
llama_rs_speculative * llama_rs_speculative_init_mtp(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float   p_min,
    bool    backend_sampling);

void    llama_rs_speculative_free(llama_rs_speculative * spec);
bool    llama_rs_speculative_need_embd_nextn(const llama_rs_speculative * spec);

llama_rs_status llama_rs_speculative_begin(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    const llama_token * prompt, size_t prompt_len);

llama_rs_status llama_rs_speculative_process(
    llama_rs_speculative * spec, const struct llama_batch * batch);

llama_rs_status llama_rs_speculative_draft(
    llama_rs_speculative * spec, llama_seq_id seq_id,
    llama_pos n_past, llama_token id_last,
    const llama_token * prompt, size_t prompt_len,
    llama_token * out_buf, size_t out_cap, size_t * out_len);

llama_rs_status llama_rs_speculative_accept(
    llama_rs_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted);

#ifdef __cplusplus
}
#endif
