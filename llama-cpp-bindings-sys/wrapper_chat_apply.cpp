#include "wrapper_chat_apply.h"
#include "wrapper_token_text.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"

#include <exception>
#include <new>
#include <nlohmann/json.hpp>
#include <string>

using wrapper_helpers::token_text_or_empty;

extern "C" llama_rs_apply_chat_template_status llama_rs_apply_chat_template(
    const struct llama_model * model,
    const char * template_src,
    const char * const * roles,
    const char * const * contents,
    size_t n_messages,
    int add_generation_prompt,
    int enable_thinking,
    char ** out_string,
    char ** out_error) {
    if (out_string) {
        *out_string = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }
    if (!model) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_MODEL_ARG;
    }
    if (!template_src) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_TEMPLATE_ARG;
    }
    if (n_messages > 0 && (!roles || !contents)) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_MESSAGES_ARG;
    }
    if (!out_string) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_OUT_STRING_ARG;
    }
    if (!out_error) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_NULL_OUT_ERROR_ARG;
    }

    try {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (!vocab) {
            return LLAMA_RS_APPLY_CHAT_TEMPLATE_MODEL_HAS_NO_VOCAB;
        }

        std::string bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template tmpl(template_src, bos_token, eos_token);

        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        for (size_t index = 0; index < n_messages; index++) {
            messages.push_back({
                { "role", roles[index] ? roles[index] : "" },
                { "content", contents[index] ? contents[index] : "" },
            });
        }

        autoparser::generation_params inputs;
        inputs.messages              = std::move(messages);
        inputs.tools                 = nlohmann::ordered_json::array();
        inputs.add_generation_prompt = add_generation_prompt != 0;
        inputs.enable_thinking       = enable_thinking != 0;

        std::string rendered = common_chat_template_direct_apply(tmpl, inputs);
        if (rendered.empty()) {
            return LLAMA_RS_APPLY_CHAT_TEMPLATE_TEMPLATE_APPLICATION_FAILED;
        }

        *out_string = llama_rs_dup_string(rendered);
        if (!*out_string) {
            return LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
        }

        return LLAMA_RS_APPLY_CHAT_TEMPLATE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));
        if (!*out_error) {
            return LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));
        if (!*out_error) {
            return LLAMA_RS_APPLY_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_APPLY_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION;
    }
}
