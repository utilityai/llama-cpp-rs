#include "wrapper_mtmd.h"
#include "wrapper_utils.h"

#include <exception>
#include <new>
#include <string>

extern "C" llama_rs_mtmd_init_from_file_status llama_rs_mtmd_init_from_file(
    const char * mmproj_path,
    const struct llama_model * text_model,
    struct mtmd_context_params ctx_params,
    struct mtmd_context ** out_ctx,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!out_ctx) {
        return LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG;
    }
    *out_ctx = nullptr;
    if (!mmproj_path) {
        return LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG;
    }
    if (!text_model) {
        return LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG;
    }

    try {
        struct mtmd_context * ctx = mtmd_init_from_file(mmproj_path, text_model, ctx_params);
        if (!ctx) {
            return LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_RETURNED_NULL;
        }
        *out_ctx = ctx;
        return LLAMA_RS_MTMD_INIT_FROM_FILE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_mtmd_bitmap_init_from_file_status llama_rs_mtmd_bitmap_init_from_file(
    struct mtmd_context * ctx,
    const char * fname,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (!out_bitmap) {
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_OUT_BITMAP_ARG;
    }
    *out_bitmap = nullptr;
    if (!ctx) {
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_CTX_ARG;
    }
    if (!fname) {
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_FNAME_ARG;
    }

    try {
        struct mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_file(ctx, fname);
        if (!bitmap) {
            return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_VENDORED_RETURNED_NULL;
        }
        *out_bitmap = bitmap;
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_mtmd_tokenize_status llama_rs_mtmd_tokenize(
    struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap ** bitmaps,
    size_t num_bitmaps,
    int32_t * out_undocumented_return_code,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_undocumented_return_code) {
        *out_undocumented_return_code = 0;
    }
    if (!ctx) {
        return LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG;
    }
    if (!output) {
        return LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG;
    }
    if (!text) {
        return LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG;
    }
    if (num_bitmaps > 0 && !bitmaps) {
        return LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO;
    }

    try {
        int32_t result = mtmd_tokenize(ctx, output, text, bitmaps, num_bitmaps);
        switch (result) {
            case 0:
                return LLAMA_RS_MTMD_TOKENIZE_OK;
            case 1:
                return LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT;
            case 2:
                return LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_IMAGE_PREPROCESSING_ERROR;
            default:
                if (out_undocumented_return_code) {
                    *out_undocumented_return_code = result;
                }
                return LLAMA_RS_MTMD_TOKENIZE_VENDORED_RETURNED_UNDOCUMENTED_NONZERO_CODE;
        }
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_TOKENIZE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_TOKENIZE_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_mtmd_encode_chunk_status llama_rs_mtmd_encode_chunk(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk,
    int32_t * out_vendored_return_code,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_vendored_return_code) {
        *out_vendored_return_code = 0;
    }
    if (!ctx) {
        return LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG;
    }
    if (!chunk) {
        return LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG;
    }

    try {
        int32_t result = mtmd_encode_chunk(ctx, chunk);
        if (result != 0) {
            if (out_vendored_return_code) {
                *out_vendored_return_code = result;
            }
            return LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_RETURNED_NONZERO_CODE;
        }
        return LLAMA_RS_MTMD_ENCODE_CHUNK_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_mtmd_eval_chunk_single_status llama_rs_mtmd_eval_chunk_single(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunk * chunk,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_n_past,
    int32_t * out_vendored_return_code,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_vendored_return_code) {
        *out_vendored_return_code = 0;
    }
    if (!ctx) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG;
    }
    if (!lctx) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG;
    }
    if (!chunk) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG;
    }
    if (!out_new_n_past) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG;
    }

    try {
        int32_t result = mtmd_helper_eval_chunk_single(
            ctx, lctx, chunk, n_past, seq_id, n_batch, logits_last, out_new_n_past);
        if (result != 0) {
            if (out_vendored_return_code) {
                *out_vendored_return_code = result;
            }
            return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_VENDORED_RETURNED_NONZERO_CODE;
        }
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_VENDORED_THREW_CXX_EXCEPTION;
    }
}
