#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
#define LLAMA_RS_NOEXCEPT noexcept
#else
#define LLAMA_RS_NOEXCEPT
#endif

typedef enum llama_rs_status {
    LLAMA_RS_STATUS_OK = 0,
    LLAMA_RS_STATUS_INVALID_ARGUMENT = -1,
    LLAMA_RS_STATUS_ALLOCATION_FAILED = -2,
    LLAMA_RS_STATUS_EXCEPTION = -3
} llama_rs_status;

#ifdef __cplusplus

#include <cstring>
#include <exception>
#include <new>
#include <string>

static inline char * llama_rs_dup_string(const std::string & value) {
    char * buffer = new (std::nothrow) char[value.size() + 1];
    if (!buffer) {
        return nullptr;
    }
    std::memcpy(buffer, value.data(), value.size());
    buffer[value.size()] = '\0';
    return buffer;
}

static inline bool llama_rs_capture_message(char ** out_error, const char * message) {
    if (out_error == nullptr) {
        return true;
    }
    *out_error = llama_rs_dup_string(message);
    return *out_error != nullptr;
}

template <typename TStatus>
static inline TStatus llama_rs_capture_exception(
    char ** out_error, TStatus allocation_failed, TStatus threw_cxx_exception) {
    try {
        throw;
    } catch (const std::bad_alloc &) {
        return allocation_failed;
    } catch (const std::exception & error) {
        return llama_rs_capture_message(out_error, error.what()) ? threw_cxx_exception
                                                                 : allocation_failed;
    } catch (...) {
        return llama_rs_capture_message(out_error, "unknown c++ exception") ? threw_cxx_exception
                                                                            : allocation_failed;
    }
}

#endif
