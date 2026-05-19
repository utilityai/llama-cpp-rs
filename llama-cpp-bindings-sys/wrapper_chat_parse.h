#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_rs_parsed_chat;
typedef struct llama_rs_parsed_chat * llama_rs_parsed_chat_handle;

typedef enum llama_rs_parse_chat_message_status {
    LLAMA_RS_PARSE_CHAT_MESSAGE_OK = 0,
    LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_MODEL_ARG,
    LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_INPUT_ARG,
    LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_HANDLE_ARG,
    LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_CHAT_TEMPLATE,
    LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_VOCAB,
    LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSE_CHAT_MESSAGE_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_parse_chat_message_status;

llama_rs_parse_chat_message_status llama_rs_parse_chat_message(
    const struct llama_model * model,
    const char * tools_json,
    const char * input,
    int is_partial,
    llama_rs_parsed_chat_handle * out_handle,
    char ** out_error);

typedef enum llama_rs_parsed_chat_free_status {
    LLAMA_RS_PARSED_CHAT_FREE_OK = 0,
    LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
} llama_rs_parsed_chat_free_status;

llama_rs_parsed_chat_free_status llama_rs_parsed_chat_free(
    llama_rs_parsed_chat_handle handle,
    char ** out_error);

typedef enum llama_rs_parsed_chat_tool_call_count_status {
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_OK = 0,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_HANDLE_ARG,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_OUT_COUNT_ARG,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_parsed_chat_tool_call_count_status;

llama_rs_parsed_chat_tool_call_count_status llama_rs_parsed_chat_tool_call_count(
    llama_rs_parsed_chat_handle handle,
    size_t * out_count,
    char ** out_error);

typedef enum llama_rs_parsed_chat_tool_call_id_status {
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_OK = 0,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_HANDLE_ARG,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_OUT_STRING_ARG,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_INDEX_OUT_OF_BOUNDS,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_parsed_chat_tool_call_id_status;

llama_rs_parsed_chat_tool_call_id_status llama_rs_parsed_chat_tool_call_id(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error);

typedef enum llama_rs_parsed_chat_tool_call_name_status {
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_OK = 0,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_HANDLE_ARG,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_OUT_STRING_ARG,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_INDEX_OUT_OF_BOUNDS,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_parsed_chat_tool_call_name_status;

llama_rs_parsed_chat_tool_call_name_status llama_rs_parsed_chat_tool_call_name(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error);

typedef enum llama_rs_parsed_chat_tool_call_arguments_status {
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_OK = 0,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_HANDLE_ARG,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_OUT_STRING_ARG,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_INDEX_OUT_OF_BOUNDS,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_parsed_chat_tool_call_arguments_status;

llama_rs_parsed_chat_tool_call_arguments_status llama_rs_parsed_chat_tool_call_arguments(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error);

typedef enum llama_rs_parsed_chat_content_status {
    LLAMA_RS_PARSED_CHAT_CONTENT_OK = 0,
    LLAMA_RS_PARSED_CHAT_CONTENT_NULL_HANDLE_ARG,
    LLAMA_RS_PARSED_CHAT_CONTENT_NULL_OUT_STRING_ARG,
    LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSED_CHAT_CONTENT_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_parsed_chat_content_status;

llama_rs_parsed_chat_content_status llama_rs_parsed_chat_content(
    llama_rs_parsed_chat_handle handle,
    char ** out_string,
    char ** out_error);

typedef enum llama_rs_parsed_chat_reasoning_content_status {
    LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_OK = 0,
    LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_HANDLE_ARG,
    LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_OUT_STRING_ARG,
    LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_parsed_chat_reasoning_content_status;

llama_rs_parsed_chat_reasoning_content_status llama_rs_parsed_chat_reasoning_content(
    llama_rs_parsed_chat_handle handle,
    char ** out_string,
    char ** out_error);

#ifdef __cplusplus
}
#endif
