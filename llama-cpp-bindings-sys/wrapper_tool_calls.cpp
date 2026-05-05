#include "wrapper_tool_calls.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"

#include <exception>
#include <string>

namespace {

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

}  // namespace

extern "C" llama_rs_status llama_rs_detect_tool_call_markers(
    const struct llama_model * model,
    char ** out_open,
    char ** out_close,
    char ** out_error) {
    if (out_open) {
        *out_open = nullptr;
    }
    if (out_close) {
        *out_close = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }

    if (!model || !out_open || !out_close || !out_error) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (!tmpl_src) {
            return LLAMA_RS_STATUS_OK;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (!vocab) {
            return LLAMA_RS_STATUS_OK;
        }

        std::string bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template tmpl(tmpl_src, bos_token, eos_token);

        autoparser::autoparser parser;
        parser.analyze_template(tmpl);

        if (parser.tools.format.section_start.empty()
            || parser.tools.format.section_end.empty()) {
            return LLAMA_RS_STATUS_OK;
        }

        char * open_dup = llama_rs_dup_string(parser.tools.format.section_start);
        char * close_dup = llama_rs_dup_string(parser.tools.format.section_end);

        if (!open_dup || !close_dup) {
            std::free(open_dup);
            std::free(close_dup);

            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }

        *out_open = open_dup;
        *out_close = close_dup;

        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));

        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));

        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
