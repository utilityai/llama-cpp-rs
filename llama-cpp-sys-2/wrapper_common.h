#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

struct llama_model;
struct llama_sampler;
struct llama_vocab;

struct llama_rs_grammar_trigger {
    int type;
    char * value;
    llama_token token;
};

struct llama_rs_chat_template_result {
    char * prompt;
    char * grammar;
    char * parser;
    char * generation_prompt;
    int chat_format;
    bool grammar_lazy;
    struct llama_rs_grammar_trigger * grammar_triggers;
    size_t grammar_triggers_count;
    char ** preserved_tokens;
    size_t preserved_tokens_count;
    char ** additional_stops;
    size_t additional_stops_count;
};

struct llama_rs_chat_msg_content_part_oaicompat {
    char * type;
    char * text;
};

struct llama_rs_tool_call_oaicompat {
    char * name;
    char * arguments;
    char * id;
};

struct llama_rs_chat_msg_oaicompat {
    char * role;
    char * content;
    struct llama_rs_chat_msg_content_part_oaicompat * content_parts;
    size_t content_parts_count;
    char * reasoning_content;
    char * tool_name;
    char * tool_call_id;
    struct llama_rs_tool_call_oaicompat * tool_calls;
    size_t tool_calls_count;
};

struct llama_rs_chat_tool_call_delta_oaicompat {
    char * name;
    char * arguments;
    char * id;
};

struct llama_rs_chat_msg_diff_oaicompat {
    char * reasoning_content_delta;
    char * content_delta;
    size_t tool_call_index;
    struct llama_rs_chat_tool_call_delta_oaicompat tool_call_delta;
};

struct llama_rs_chat_parse_state_oaicompat;

struct llama_rs_chat_template_oaicompat_params {
    const char * messages;
    const char * tools;
    const char * tool_choice;
    const char * json_schema;
    const char * grammar;
    const char * reasoning_format;
    const char * chat_template_kwargs;
    bool add_generation_prompt;
    bool use_jinja;
    bool parallel_tool_calls;
    bool enable_thinking;
    bool add_bos;
    bool add_eos;
};

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

llama_rs_status llama_rs_apply_chat_template_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_rs_chat_template_oaicompat_params * params,
    struct llama_rs_chat_template_result * out_result);

llama_rs_status llama_rs_chat_parse_to_oaicompat(
    const char * input,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    const char * generation_prompt,
    char ** out_json);

struct llama_rs_chat_parse_state_oaicompat * llama_rs_chat_parse_state_init_oaicompat(
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    const char * generation_prompt);

llama_rs_status llama_rs_chat_parse_state_update_oaicompat(
    struct llama_rs_chat_parse_state_oaicompat * state,
    const char * text_added,
    bool is_partial,
    struct llama_rs_chat_msg_oaicompat * out_msg,
    struct llama_rs_chat_msg_diff_oaicompat ** out_diffs,
    size_t * out_diffs_count);

llama_rs_status llama_rs_chat_msg_diff_to_oaicompat_json(
    const struct llama_rs_chat_msg_diff_oaicompat * diff,
    char ** out_json);

void llama_rs_chat_template_result_free(struct llama_rs_chat_template_result * result);
void llama_rs_chat_msg_free_oaicompat(struct llama_rs_chat_msg_oaicompat * msg);
void llama_rs_chat_msg_diff_free_oaicompat(struct llama_rs_chat_msg_diff_oaicompat * diffs, size_t count);
void llama_rs_chat_parse_state_free_oaicompat(struct llama_rs_chat_parse_state_oaicompat * state);

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

#ifdef __cplusplus
}
#endif
