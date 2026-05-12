#pragma once

#include "llama.cpp/include/llama.h"

#include <string>

namespace wrapper_helpers {

std::string token_text_or_empty(const llama_vocab * vocab, llama_token token);

}
