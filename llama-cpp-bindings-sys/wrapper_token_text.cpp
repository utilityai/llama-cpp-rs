#include "wrapper_token_text.h"
#include "llama.h"
#include <string>

namespace wrapper_helpers {

auto token_text_or_empty(const llama_vocab * vocab, llama_token token) -> std::string {
    if (token == LLAMA_TOKEN_NULL) {
        return {};
    }

    const char * text = llama_vocab_get_text(vocab, token);
    if (text == nullptr) {
        return {};
    }

    return {text};
}

}  // namespace wrapper_helpers
