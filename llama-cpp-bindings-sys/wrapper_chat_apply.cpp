#include "wrapper_chat_apply.h"
#include "nlohmann/json_fwd.hpp"
#include "wrapper_token_text.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <exception>
#include <gsl/span>
#include <new>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

using wrapper_helpers::token_text_or_empty;

extern "C" auto llama_rs_apply_chat_template(
    const struct llama_model * model,
    const char * template_src,
    const char * const * roles,
    const char * const * contents,
    size_t n_messages,
    int add_generation_prompt,
    int enable_thinking,
    char ** out_string,
    char ** out_error) -> llama_rs_apply_chat_template_status {
    if (out_string != nullptr) {
        *out_string = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_MODEL_ARG;
    }
    if (template_src == nullptr) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_TEMPLATE_ARG;
    }
    if (n_messages > 0 && ((roles == nullptr) || (contents == nullptr))) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_MESSAGES_ARG;
    }
    if (out_string == nullptr) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_OUT_STRING_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_OUT_ERROR_ARG;
    }

    try {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            return LLAMA_RS_APPLY_CHAT_TEMPLATE_MODEL_HAS_NO_VOCAB;
        }

        std::string const bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string const eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template const tmpl(template_src, bos_token, eos_token);

        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        const gsl::span<const char * const> role_span(roles, n_messages);
        const gsl::span<const char * const> content_span(contents, n_messages);
        for (size_t index = 0; index < n_messages; index++) {
            messages.push_back({
                { "role", (role_span[index] != nullptr) ? role_span[index] : "" },
                { "content", (content_span[index] != nullptr) ? content_span[index] : "" },
            });
        }

        autoparser::generation_params inputs;
        inputs.messages              = std::move(messages);
        inputs.tools                 = nlohmann::ordered_json::array();
        inputs.add_generation_prompt = add_generation_prompt != 0;
        inputs.enable_thinking       = enable_thinking != 0;

        std::string const rendered = common_chat_template_direct_apply(tmpl, inputs);
        if (rendered.empty()) {
            return LLAMA_RS_APPLY_CHAT_TEMPLATE_TEMPLATE_APPLICATION_FAILED;
        }

        *out_string = llama_rs_dup_string(rendered);
        if (*out_string == nullptr) {
            return LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
        }

        return LLAMA_RS_APPLY_CHAT_TEMPLATE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_APPLY_CHAT_TEMPLATE_THREW_CXX_EXCEPTION);
    }
}
