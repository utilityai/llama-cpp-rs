#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_rs_parsed_chat;
typedef struct llama_rs_parsed_chat * llama_rs_parsed_chat_handle;

/**
 * Parse a chat-completion turn from raw assistant output using llama.cpp's
 * `common_chat_parse`, driven by the model's autoparser-built peg parser.
 *
 * `tools_json` is a serialized JSON array of OpenAI-style tool definitions
 * (or empty / null when the request had no tools). `is_partial` switches
 * between mid-stream parses (partial accepts incomplete payloads) and final
 * parses (rejects malformed input).
 *
 * On success, `*out_handle` owns the parsed message; free via
 * `llama_rs_parsed_chat_free`. On failure, `*out_error` carries an
 * exception message; free via `llama_rs_string_free`.
 */
llama_rs_status llama_rs_parse_chat_message(
    const struct llama_model * model,
    const char * tools_json,
    const char * input,
    int is_partial,
    llama_rs_parsed_chat_handle * out_handle,
    char ** out_error);

void llama_rs_parsed_chat_free(llama_rs_parsed_chat_handle handle);

size_t llama_rs_parsed_chat_tool_call_count(llama_rs_parsed_chat_handle handle);

/**
 * Returns a heap-allocated UTF-8 string for the i-th tool call's `id`,
 * `name`, or `arguments` field. Free with `llama_rs_string_free`. Returns
 * nullptr if `handle` is null or `index` is out of bounds.
 *
 * `arguments` is the raw JSON string emitted by the parser — the caller is
 * expected to feed it into a schema validator or hand it back to clients
 * verbatim.
 */
char * llama_rs_parsed_chat_tool_call_id(llama_rs_parsed_chat_handle handle, size_t index);
char * llama_rs_parsed_chat_tool_call_name(llama_rs_parsed_chat_handle handle, size_t index);
char * llama_rs_parsed_chat_tool_call_arguments(llama_rs_parsed_chat_handle handle, size_t index);

char * llama_rs_parsed_chat_content(llama_rs_parsed_chat_handle handle);
char * llama_rs_parsed_chat_reasoning_content(llama_rs_parsed_chat_handle handle);

#ifdef __cplusplus
}
#endif
