#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_model;
struct llama_chat_message;
struct llama_sampler;
struct llama_vocab;

struct llama_rs_grammar_trigger {
    int type;
    char * value;
    llama_token token;
};

struct llama_rs_tool_call {
    char * name;
    char * arguments;
    char * id;
};

struct llama_rs_chat_msg {
    char * role;
    char * content;
    char * reasoning_content;
    char * tool_name;
    char * tool_call_id;
    struct llama_rs_tool_call * tool_calls;
    size_t tool_calls_count;
};

struct llama_rs_chat_tool_call_delta {
    char * name;
    char * arguments;
    char * id;
};

struct llama_rs_chat_msg_diff {
    char * reasoning_content_delta;
    char * content_delta;
    size_t tool_call_index;
    struct llama_rs_chat_tool_call_delta tool_call_delta;
};

struct llama_rs_chat_parse_state;

struct llama_rs_chat_template_result {
    char * prompt;
    char * grammar;
    char * parser;
    int chat_format;
    bool thinking_forced_open;
    bool grammar_lazy;
    struct llama_rs_grammar_trigger * grammar_triggers;
    size_t grammar_triggers_count;
    char ** preserved_tokens;
    size_t preserved_tokens_count;
    char ** additional_stops;
    size_t additional_stops_count;
};

int llama_rs_json_schema_to_grammar(const char * schema_json, bool force_gbnf, char ** out_grammar);

int llama_rs_apply_chat_template_with_tools(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_chat_message * messages,
    size_t message_count,
    const char * tools_json,
    const char * json_schema,
    bool add_generation_prompt,
    struct llama_rs_chat_template_result * out_result);

int llama_rs_chat_parse_to_oaicompat(
    const char * input,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    bool thinking_forced_open,
    char ** out_json);

int llama_rs_chat_parse(
    const char * input,
    bool is_partial,
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    bool thinking_forced_open,
    struct llama_rs_chat_msg * out_msg);

struct llama_rs_chat_parse_state * llama_rs_chat_parse_state_init(
    int chat_format,
    bool parse_tool_calls,
    const char * parser_data,
    bool thinking_forced_open);

int llama_rs_chat_parse_state_update(
    struct llama_rs_chat_parse_state * state,
    const char * text_added,
    bool is_partial,
    struct llama_rs_chat_msg * out_msg,
    struct llama_rs_chat_msg_diff ** out_diffs,
    size_t * out_diffs_count);

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

int llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

void llama_rs_chat_template_result_free(struct llama_rs_chat_template_result * result);
void llama_rs_chat_msg_free(struct llama_rs_chat_msg * msg);
void llama_rs_chat_msg_diff_free(struct llama_rs_chat_msg_diff * diffs, size_t count);
void llama_rs_chat_parse_state_free(struct llama_rs_chat_parse_state * state);
void llama_rs_string_free(char * ptr);

#ifdef __cplusplus
}
#endif
