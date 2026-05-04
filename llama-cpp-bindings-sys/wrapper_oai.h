#pragma once

#include "wrapper_common.h"

#ifdef __cplusplus
extern "C" {
#endif

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

llama_rs_status llama_rs_apply_chat_template_with_tools_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_chat_message * messages,
    size_t message_count,
    const char * tools_json,
    const char * json_schema,
    bool add_generation_prompt,
    char ** out_json);

llama_rs_status llama_rs_apply_chat_template_oaicompat(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_rs_chat_template_oaicompat_params * params,
    char ** out_json);

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
    char ** out_diffs_json);

void llama_rs_chat_parse_state_free_oaicompat(struct llama_rs_chat_parse_state_oaicompat * state);

#ifdef __cplusplus
}
#endif
