#pragma once

#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/include/llama.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_fit_params_status {
    LLAMA_RS_FIT_PARAMS_OK = 0,
    LLAMA_RS_FIT_PARAMS_VENDORED_REPORTED_FAILURE,
    LLAMA_RS_FIT_PARAMS_VENDORED_REPORTED_ERROR,
    LLAMA_RS_FIT_PARAMS_VENDORED_RETURNED_UNRECOGNIZED_STATUS_CODE,
    LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_FIT_PARAMS_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_fit_params_status;

llama_rs_fit_params_status llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level,
    int32_t * out_unrecognized_status_code,
    char ** out_error);

#ifdef __cplusplus
}
#endif
