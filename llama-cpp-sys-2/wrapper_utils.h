#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

static inline char * llama_rs_dup_string(const std::string & value) {
    char * buffer = static_cast<char *>(std::malloc(value.size() + 1));
    if (!buffer) {
        return nullptr;
    }
    std::memcpy(buffer, value.data(), value.size());
    buffer[value.size()] = '\0';
    return buffer;
}
