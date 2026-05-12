#include "wrapper_token_text.h"

namespace wrapper_helpers {

std::string token_text_or_empty(const llama_vocab * vocab, llama_token token) {
    if (token == LLAMA_TOKEN_NULL) {
        return {};
    }

    const char * text = llama_vocab_get_text(vocab, token);
    if (!text) {
        return {};
    }

    return std::string(text);
}

}
