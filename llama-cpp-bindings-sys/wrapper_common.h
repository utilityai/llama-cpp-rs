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

typedef enum llama_rs_json_schema_to_grammar_status {
    LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK = 0,
    LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_SCHEMA_JSON_ARG,
    LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_GRAMMAR_ARG,
    LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_ERROR_ARG,
    LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_SCHEMA,
    LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_THREW_CXX_EXCEPTION,
} llama_rs_json_schema_to_grammar_status;

llama_rs_json_schema_to_grammar_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar,
    char ** out_error);

typedef enum llama_rs_sampler_init_grammar_status {
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_OK = 0,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_SAMPLER_ARG,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_NULL_OUT_ERROR_ARG,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_COMPILATION_FAILED,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_THREW_CXX_EXCEPTION,
} llama_rs_sampler_init_grammar_status;

llama_rs_sampler_init_grammar_status llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    struct llama_sampler ** out_sampler,
    char ** out_error);

typedef enum llama_rs_sampler_init_grammar_lazy_status {
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_OK = 0,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_NULL_OUT_SAMPLER_ARG,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_NULL_OUT_ERROR_ARG,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_COMPILATION_FAILED,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_THREW_CXX_EXCEPTION,
} llama_rs_sampler_init_grammar_lazy_status;

llama_rs_sampler_init_grammar_lazy_status llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    struct llama_sampler ** out_sampler,
    char ** out_error);

typedef enum llama_rs_sampler_init_grammar_lazy_patterns_status {
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_OK = 0,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_SAMPLER_ARG,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_NULL_OUT_ERROR_ARG,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_COMPILATION_FAILED,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_INVALID_TRIGGER_PATTERN,
    LLAMA_RS_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_THREW_CXX_EXCEPTION,
} llama_rs_sampler_init_grammar_lazy_patterns_status;

llama_rs_sampler_init_grammar_lazy_patterns_status llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens,
    struct llama_sampler ** out_sampler,
    char ** out_error);

typedef enum llama_rs_sampler_accept_status {
    LLAMA_RS_SAMPLER_ACCEPT_OK = 0,
    LLAMA_RS_SAMPLER_ACCEPT_NULL_SAMPLER_ARG,
    LLAMA_RS_SAMPLER_ACCEPT_NULL_OUT_ERROR_ARG,
    LLAMA_RS_SAMPLER_ACCEPT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_SAMPLER_ACCEPT_THREW_CXX_EXCEPTION,
} llama_rs_sampler_accept_status;

llama_rs_sampler_accept_status llama_rs_sampler_accept(
    struct llama_sampler * sampler,
    llama_token token,
    char ** out_error);

typedef enum llama_rs_sampler_sample_status {
    LLAMA_RS_SAMPLER_SAMPLE_OK = 0,
    LLAMA_RS_SAMPLER_SAMPLE_NULL_SAMPLER_ARG,
    LLAMA_RS_SAMPLER_SAMPLE_NULL_CTX_ARG,
    LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_TOKEN_ARG,
    LLAMA_RS_SAMPLER_SAMPLE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_SAMPLER_SAMPLE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_SAMPLER_SAMPLE_THREW_CXX_EXCEPTION,
} llama_rs_sampler_sample_status;

llama_rs_sampler_sample_status llama_rs_sampler_sample(
    struct llama_sampler * sampler,
    struct llama_context * ctx,
    int32_t idx,
    llama_token * out_token,
    char ** out_error);

void llama_rs_string_free(char * ptr);

llama_pos llama_rs_memory_seq_pos_max(
    const struct llama_context * ctx,
    llama_seq_id seq_id) LLAMA_RS_NOEXCEPT;

typedef enum llama_rs_encode_status {
    LLAMA_RS_ENCODE_OK = 0,
    LLAMA_RS_ENCODE_NULL_CTX_ARG,
    LLAMA_RS_ENCODE_MODEL_HAS_NO_ENCODER,
    LLAMA_RS_ENCODE_RETURNED_ERROR_CODE,
    LLAMA_RS_ENCODE_OUT_OF_MEMORY,
    LLAMA_RS_ENCODE_COMPUTE_FAILED,
    LLAMA_RS_ENCODE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_ENCODE_THREW_CXX_EXCEPTION,
} llama_rs_encode_status;

llama_rs_encode_status llama_rs_encode(
    struct llama_context * ctx,
    struct llama_batch batch,
    int32_t * out_return_code,
    char ** out_error);

typedef enum llama_rs_memory_seq_add_status {
    LLAMA_RS_MEMORY_SEQ_ADD_OK = 0,
    LLAMA_RS_MEMORY_SEQ_ADD_NULL_CTX_ARG,
    LLAMA_RS_MEMORY_SEQ_ADD_INCOMPATIBLE_ROPE_TYPE,
    LLAMA_RS_MEMORY_SEQ_ADD_NULL_MEM,
    LLAMA_RS_MEMORY_SEQ_ADD_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MEMORY_SEQ_ADD_THREW_CXX_EXCEPTION,
} llama_rs_memory_seq_add_status;

llama_rs_memory_seq_add_status llama_rs_memory_seq_add(
    const struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos_start,
    llama_pos pos_end,
    llama_pos shift,
    char ** out_error);

typedef enum llama_rs_memory_seq_div_status {
    LLAMA_RS_MEMORY_SEQ_DIV_OK = 0,
    LLAMA_RS_MEMORY_SEQ_DIV_NULL_CTX_ARG,
    LLAMA_RS_MEMORY_SEQ_DIV_INCOMPATIBLE_ROPE_TYPE,
    LLAMA_RS_MEMORY_SEQ_DIV_NULL_MEM,
    LLAMA_RS_MEMORY_SEQ_DIV_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MEMORY_SEQ_DIV_THREW_CXX_EXCEPTION,
} llama_rs_memory_seq_div_status;

llama_rs_memory_seq_div_status llama_rs_memory_seq_div(
    const struct llama_context * ctx,
    llama_seq_id seq_id,
    llama_pos pos_start,
    llama_pos pos_end,
    int divisor,
    char ** out_error);

typedef enum llama_rs_load_model_from_file_status {
    LLAMA_RS_LOAD_MODEL_FROM_FILE_OK = 0,
    LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_PATH_ARG,
    LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_MODEL_ARG,
    LLAMA_RS_LOAD_MODEL_FROM_FILE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_LOAD_MODEL_FROM_FILE_LOAD_FAILED,
    LLAMA_RS_LOAD_MODEL_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_LOAD_MODEL_FROM_FILE_THREW_CXX_EXCEPTION,
} llama_rs_load_model_from_file_status;

llama_rs_load_model_from_file_status llama_rs_load_model_from_file(
    const char * path,
    struct llama_model_params params,
    struct llama_model ** out_model,
    char ** out_error);

typedef enum llama_rs_new_context_with_model_status {
    LLAMA_RS_NEW_CONTEXT_WITH_MODEL_OK = 0,
    LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_MODEL_ARG,
    LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_OUT_CTX_ARG,
    LLAMA_RS_NEW_CONTEXT_WITH_MODEL_NULL_OUT_ERROR_ARG,
    LLAMA_RS_NEW_CONTEXT_WITH_MODEL_CREATION_FAILED,
    LLAMA_RS_NEW_CONTEXT_WITH_MODEL_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_NEW_CONTEXT_WITH_MODEL_THREW_CXX_EXCEPTION,
} llama_rs_new_context_with_model_status;

llama_rs_new_context_with_model_status llama_rs_new_context_with_model(
    struct llama_model * model,
    struct llama_context_params params,
    struct llama_context ** out_ctx,
    char ** out_error);

typedef enum llama_rs_decode_status {
    LLAMA_RS_DECODE_OK = 0,
    LLAMA_RS_DECODE_NULL_CTX_ARG,
    LLAMA_RS_DECODE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_DECODE_RETURNED_ERROR_CODE,
    LLAMA_RS_DECODE_OUT_OF_MEMORY,
    LLAMA_RS_DECODE_COMPUTE_FAILED,
    LLAMA_RS_DECODE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_DECODE_THREW_CXX_EXCEPTION,
} llama_rs_decode_status;

llama_rs_decode_status llama_rs_decode(
    struct llama_context * ctx,
    struct llama_batch batch,
    int32_t * out_return_code,
    char ** out_error);

typedef enum llama_rs_tokenize_status {
    LLAMA_RS_TOKENIZE_OK = 0,
    LLAMA_RS_TOKENIZE_NULL_VOCAB_ARG,
    LLAMA_RS_TOKENIZE_NULL_TEXT_ARG,
    LLAMA_RS_TOKENIZE_NULL_OUT_RETURNED_COUNT_ARG,
    LLAMA_RS_TOKENIZE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_TOKENIZE_THREW_CXX_EXCEPTION,
} llama_rs_tokenize_status;

llama_rs_tokenize_status llama_rs_tokenize(
    const struct llama_vocab * vocab,
    const char * text,
    int32_t text_len,
    llama_token * tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special,
    int32_t * out_returned_count,
    char ** out_error);

typedef enum llama_rs_sampler_apply_status {
    LLAMA_RS_SAMPLER_APPLY_OK = 0,
    LLAMA_RS_SAMPLER_APPLY_NULL_SAMPLER_ARG,
    LLAMA_RS_SAMPLER_APPLY_NULL_DATA_ARRAY_ARG,
    LLAMA_RS_SAMPLER_APPLY_NULL_OUT_ERROR_ARG,
    LLAMA_RS_SAMPLER_APPLY_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_SAMPLER_APPLY_THREW_CXX_EXCEPTION,
} llama_rs_sampler_apply_status;

llama_rs_sampler_apply_status llama_rs_sampler_apply(
    struct llama_sampler * sampler,
    struct llama_token_data_array * data_array,
    char ** out_error);

#ifdef __cplusplus
}
#endif
