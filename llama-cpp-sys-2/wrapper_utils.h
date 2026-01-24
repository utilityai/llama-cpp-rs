#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef enum llama_rs_status {
    LLAMA_RS_STATUS_OK = 0,
    LLAMA_RS_STATUS_INVALID_ARGUMENT = -1,
    LLAMA_RS_STATUS_ALLOCATION_FAILED = -2,
    LLAMA_RS_STATUS_EXCEPTION = -3
} llama_rs_status;

#ifdef __cplusplus

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.cpp/common/common.h"

static inline char * llama_rs_dup_string(const std::string & value) {
    char * buffer = static_cast<char *>(std::malloc(value.size() + 1));
    if (!buffer) {
        return nullptr;
    }
    std::memcpy(buffer, value.data(), value.size());
    buffer[value.size()] = '\0';
    return buffer;
}

static inline llama_rs_status dup_string_array(
    const std::vector<std::string> & values,
    char *** out_items,
    size_t * out_count) {
    if (!out_items || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_items = nullptr;
    *out_count = 0;
    if (values.empty()) {
        return LLAMA_RS_STATUS_OK;
    }

    char ** items =
        static_cast<char **>(std::malloc(sizeof(char *) * values.size()));
    if (!items) {
        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        items[i] = llama_rs_dup_string(values[i]);
        if (!items[i]) {
            for (size_t j = 0; j < i; ++j) {
                std::free(items[j]);
            }
            std::free(items);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    *out_items = items;
    *out_count = values.size();
    return LLAMA_RS_STATUS_OK;
}

static inline llama_rs_status dup_trigger_array(
    const std::vector<common_grammar_trigger> & triggers,
    struct llama_rs_grammar_trigger ** out_items,
    size_t * out_count) {
    if (!out_items || !out_count) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_items = nullptr;
    *out_count = 0;
    if (triggers.empty()) {
        return LLAMA_RS_STATUS_OK;
    }

    auto * items = static_cast<struct llama_rs_grammar_trigger *>(
        std::malloc(sizeof(struct llama_rs_grammar_trigger) * triggers.size()));
    if (!items) {
        return LLAMA_RS_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < triggers.size(); ++i) {
        items[i].type = static_cast<int>(triggers[i].type);
        items[i].token = triggers[i].token;
        items[i].value = llama_rs_dup_string(triggers[i].value);
        if (!items[i].value && !triggers[i].value.empty()) {
            for (size_t j = 0; j < i; ++j) {
                std::free(items[j].value);
            }
            std::free(items);
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
    }
    *out_items = items;
    *out_count = triggers.size();
    return LLAMA_RS_STATUS_OK;
}

#endif
