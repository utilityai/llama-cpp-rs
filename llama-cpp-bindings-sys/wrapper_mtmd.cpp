#include "wrapper_mtmd.h"
#include "llama.h"
#include "tools/mtmd/mtmd.h"
#include "tools/mtmd/mtmd-helper.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>

extern "C" auto llama_rs_mtmd_init_from_file(
    const char * mmproj_path,
    const struct llama_model * text_model,
    struct mtmd_context_params ctx_params,
    struct mtmd_context ** out_ctx,
    char ** out_error) -> llama_rs_mtmd_init_from_file_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_ctx == nullptr) {
        return LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG;
    }
    *out_ctx = nullptr;
    if (mmproj_path == nullptr) {
        return LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG;
    }
    if (text_model == nullptr) {
        return LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG;
    }

    try {
        struct mtmd_context * ctx = mtmd_init_from_file(mmproj_path, text_model, ctx_params);
        if (ctx == nullptr) {
            return LLAMA_RS_MTMD_INIT_FROM_FILE_INITIALIZATION_FAILED;
        }
        *out_ctx = ctx;
        return LLAMA_RS_MTMD_INIT_FROM_FILE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_MTMD_INIT_FROM_FILE_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_mtmd_bitmap_init_from_file(
    struct mtmd_context * ctx,
    const char * fname,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) -> llama_rs_mtmd_bitmap_init_from_file_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_bitmap == nullptr) {
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_OUT_BITMAP_ARG;
    }
    *out_bitmap = nullptr;
    if (ctx == nullptr) {
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_CTX_ARG;
    }
    if (fname == nullptr) {
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_FNAME_ARG;
    }

    try {
        struct mtmd_helper_bitmap_wrapper const bitmap_wrapper =
            mtmd_helper_bitmap_init_from_file(ctx, fname, false);
        struct mtmd_bitmap * bitmap = bitmap_wrapper.bitmap;
        if (bitmap == nullptr) {
            return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LOAD_FAILED;
        }
        *out_bitmap = bitmap;
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_mtmd_tokenize(
    struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap ** bitmaps,
    size_t num_bitmaps,
    int32_t * out_undocumented_return_code,
    char ** out_error) -> llama_rs_mtmd_tokenize_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_undocumented_return_code != nullptr) {
        *out_undocumented_return_code = 0;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG;
    }
    if (output == nullptr) {
        return LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG;
    }
    if (text == nullptr) {
        return LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG;
    }
    if (num_bitmaps > 0 && (bitmaps == nullptr)) {
        return LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO;
    }

    try {
        int32_t const result = mtmd_tokenize(ctx, output, text, bitmaps, num_bitmaps);
        switch (result) {
            case 0:
                return LLAMA_RS_MTMD_TOKENIZE_OK;
            case 1:
                return LLAMA_RS_MTMD_TOKENIZE_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT;
            case 2:
                return LLAMA_RS_MTMD_TOKENIZE_IMAGE_PREPROCESSING_ERROR;
            default:
                if (out_undocumented_return_code != nullptr) {
                    *out_undocumented_return_code = result;
                }
                return LLAMA_RS_MTMD_TOKENIZE_UNDOCUMENTED_ERROR_CODE;
        }
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_MTMD_TOKENIZE_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_mtmd_encode_chunk(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk,
    int32_t * out_return_code,
    char ** out_error) -> llama_rs_mtmd_encode_chunk_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_return_code != nullptr) {
        *out_return_code = 0;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG;
    }
    if (chunk == nullptr) {
        return LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG;
    }

    try {
        int32_t const result = mtmd_encode_chunk(ctx, chunk);
        if (result != 0) {
            if (out_return_code != nullptr) {
                *out_return_code = result;
            }
            return LLAMA_RS_MTMD_ENCODE_CHUNK_RETURNED_ERROR_CODE;
        }
        return LLAMA_RS_MTMD_ENCODE_CHUNK_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_MTMD_ENCODE_CHUNK_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_mtmd_eval_chunk_single(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunk * chunk,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_n_past,
    int32_t * out_return_code,
    char ** out_error) -> llama_rs_mtmd_eval_chunk_single_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_return_code != nullptr) {
        *out_return_code = 0;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG;
    }
    if (lctx == nullptr) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG;
    }
    if (chunk == nullptr) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG;
    }
    if (out_new_n_past == nullptr) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG;
    }

    try {
        int32_t const result = mtmd_helper_eval_chunk_single(
            ctx, lctx, chunk, n_past, seq_id, n_batch, logits_last, out_new_n_past);
        if (result != 0) {
            if (out_return_code != nullptr) {
                *out_return_code = result;
            }
            return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_RETURNED_ERROR_CODE;
        }
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_THREW_CXX_EXCEPTION);
    }
}
