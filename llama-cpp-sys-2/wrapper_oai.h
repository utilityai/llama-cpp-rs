#pragma once

#include "wrapper_common.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_rs_chat_msg_content_part {
    char * type;
    char * text;
};

struct llama_rs_tool_call {
    char * name;
    char * arguments;
    char * id;
};

struct llama_rs_chat_msg {
    char * role;
    char * content;
    struct llama_rs_chat_msg_content_part * content_parts;
    size_t content_parts_count;
    char * reasoning_content;
    char * tool_name;
    char * tool_call_id;
    struct llama_rs_tool_call * tool_calls;
    size_t tool_calls_count;
};

struct llama_rs_chat_tool {
    char * name;
    char * description;
    char * parameters;
};

enum llama_rs_chat_tool_choice {
    LLAMA_RS_CHAT_TOOL_CHOICE_AUTO = 0,
    LLAMA_RS_CHAT_TOOL_CHOICE_REQUIRED = 1,
    LLAMA_RS_CHAT_TOOL_CHOICE_NONE = 2,
};

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

llama_rs_status llama_rs_apply_chat_template_with_tools(
    const struct llama_model * model,
    const char * chat_template,
    const struct llama_chat_message * messages,
    size_t message_count,
    const char * tools_json,
    const char * json_schema,
    bool add_generation_prompt,
    struct llama_rs_chat_template_result * out_result);

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
    bool thinking_forced_open,
    char ** out_json);

llama_rs_status llama_rs_chat_tools_parse_oaicompat(
    const char * tools_json,
    struct llama_rs_chat_tool ** out_tools,
    size_t * out_count);

llama_rs_status llama_rs_chat_tools_to_oaicompat_json(
    const struct llama_rs_chat_tool * tools,
    size_t tools_count,
    char ** out_json);

llama_rs_status llama_rs_chat_msgs_parse_oaicompat(
    const char * messages_json,
    struct llama_rs_chat_msg ** out_msgs,
    size_t * out_count);

llama_rs_status llama_rs_chat_msgs_to_oaicompat_json(
    const struct llama_rs_chat_msg * messages,
    size_t messages_count,
    bool concat_typed_text,
    char ** out_json);

llama_rs_status llama_rs_chat_tool_choice_parse_oaicompat(
    const char * tool_choice,
    enum llama_rs_chat_tool_choice * out_choice);

void llama_rs_chat_tools_free(struct llama_rs_chat_tool * tools, size_t count);
void llama_rs_chat_msgs_free(struct llama_rs_chat_msg * msgs, size_t count);
void llama_rs_chat_msg_free(struct llama_rs_chat_msg * msg);

#ifdef __cplusplus
}
#endif
