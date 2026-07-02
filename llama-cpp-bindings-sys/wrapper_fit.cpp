#include "wrapper_fit.h"
#include "llama.h"
#include "ggml.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>

#include "llama.cpp/common/fit.h"

extern "C" auto llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level,
    int32_t * out_unrecognized_status_code,
    char ** out_error) -> llama_rs_fit_params_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_unrecognized_status_code != nullptr) {
        *out_unrecognized_status_code = 0;
    }

    try {
        const common_params_fit_status status = common_fit_params(
            path_model, mparams, cparams, tensor_split, tensor_buft_overrides,
            margins, n_ctx_min, log_level);
        switch (status) {
            case COMMON_PARAMS_FIT_STATUS_SUCCESS:
                return LLAMA_RS_FIT_PARAMS_OK;
            case COMMON_PARAMS_FIT_STATUS_FAILURE:
                return LLAMA_RS_FIT_PARAMS_DOES_NOT_FIT;
            case COMMON_PARAMS_FIT_STATUS_ERROR:
                return LLAMA_RS_FIT_PARAMS_HARD_ERROR;
        }
        if (out_unrecognized_status_code != nullptr) {
            *out_unrecognized_status_code = static_cast<int32_t>(status);
        }
        return LLAMA_RS_FIT_PARAMS_UNRECOGNIZED_STATUS_CODE;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_FIT_PARAMS_THREW_CXX_EXCEPTION);
    }
}
