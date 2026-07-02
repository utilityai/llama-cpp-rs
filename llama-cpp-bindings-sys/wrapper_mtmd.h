#pragma once

#include "llama.cpp/include/llama.h"
#include "llama.cpp/tools/mtmd/mtmd.h"
#include "llama.cpp/tools/mtmd/mtmd-helper.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_mtmd_init_from_file_status {
    LLAMA_RS_MTMD_INIT_FROM_FILE_OK = 0,
    LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG,
    LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG,
    LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG,
    LLAMA_RS_MTMD_INIT_FROM_FILE_INITIALIZATION_FAILED,
    LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_INIT_FROM_FILE_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_init_from_file_status;

llama_rs_mtmd_init_from_file_status llama_rs_mtmd_init_from_file(
    const char * mmproj_path,
    const struct llama_model * text_model,
    struct mtmd_context_params ctx_params,
    struct mtmd_context ** out_ctx,
    char ** out_error);

typedef enum llama_rs_mtmd_bitmap_init_from_file_status {
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK = 0,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_CTX_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_FNAME_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_OUT_BITMAP_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LOAD_FAILED,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_bitmap_init_from_file_status;

llama_rs_mtmd_bitmap_init_from_file_status llama_rs_mtmd_bitmap_init_from_file(
    struct mtmd_context * ctx,
    const char * fname,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

typedef enum llama_rs_mtmd_tokenize_status {
    LLAMA_RS_MTMD_TOKENIZE_OK = 0,
    LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG,
    LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG,
    LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG,
    LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO,
    LLAMA_RS_MTMD_TOKENIZE_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT,
    LLAMA_RS_MTMD_TOKENIZE_IMAGE_PREPROCESSING_ERROR,
    LLAMA_RS_MTMD_TOKENIZE_UNDOCUMENTED_ERROR_CODE,
    LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_TOKENIZE_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_tokenize_status;

llama_rs_mtmd_tokenize_status llama_rs_mtmd_tokenize(
    struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap ** bitmaps,
    size_t num_bitmaps,
    int32_t * out_undocumented_return_code,
    char ** out_error);

typedef enum llama_rs_mtmd_encode_chunk_status {
    LLAMA_RS_MTMD_ENCODE_CHUNK_OK = 0,
    LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG,
    LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG,
    LLAMA_RS_MTMD_ENCODE_CHUNK_RETURNED_ERROR_CODE,
    LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_ENCODE_CHUNK_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_encode_chunk_status;

llama_rs_mtmd_encode_chunk_status llama_rs_mtmd_encode_chunk(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk,
    int32_t * out_return_code,
    char ** out_error);

typedef enum llama_rs_mtmd_eval_chunk_single_status {
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK = 0,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_RETURNED_ERROR_CODE,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_eval_chunk_single_status;

llama_rs_mtmd_eval_chunk_single_status llama_rs_mtmd_eval_chunk_single(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunk * chunk,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_n_past,
    int32_t * out_return_code,
    char ** out_error);

#ifdef __cplusplus
}
#endif
