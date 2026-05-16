#include "wrapper_chat_parse.h"
#include "wrapper_token_text.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"
#include "marker_probes/marker_probe.h"

#include <exception>
#include <new>
#include <nlohmann/json.hpp>
#include <string>

using wrapper_helpers::token_text_or_empty;

struct llama_rs_parsed_chat {
    common_chat_msg message;
};

static char * dup_or_set_alloc_flag(const std::string & source, bool * out_alloc_failed) {
    *out_alloc_failed = false;
    char * dup = llama_rs_dup_string(source);
    if (!dup) {
        *out_alloc_failed = true;
    }
    return dup;
}

extern "C" llama_rs_parse_chat_message_status llama_rs_parse_chat_message(
    const struct llama_model * model,
    const char * tools_json,
    const char * input,
    int is_partial,
    llama_rs_parsed_chat_handle * out_handle,
    char ** out_error) {
    if (out_handle) {
        *out_handle = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }
    if (!model) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_MODEL_ARG;
    }
    if (!input) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_INPUT_ARG;
    }
    if (!out_handle) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_HANDLE_ARG;
    }
    if (!out_error) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_ERROR_ARG;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (!tmpl_src) {
            return LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_CHAT_TEMPLATE;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (!vocab) {
            return LLAMA_RS_PARSE_CHAT_MESSAGE_MODEL_HAS_NO_VOCAB;
        }

        std::string bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template tmpl(tmpl_src, bos_token, eos_token);

        autoparser::autoparser parser;
        parser.analyze_template(tmpl);

        if (parser.reasoning.mode == autoparser::reasoning_mode::NONE) {
            for (auto probe : marker_probes::registered()) {
                auto fallback = probe(tmpl);
                if (fallback.found) {
                    parser.reasoning.mode  = autoparser::reasoning_mode::TAG_BASED;
                    parser.reasoning.start = std::move(fallback.start);
                    parser.reasoning.end   = std::move(fallback.end);
                    break;
                }
            }
        }

        autoparser::generation_params inputs;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = true;
        inputs.messages = nlohmann::ordered_json::array({
            { { "role", "user" }, { "content", "ping" } }
        });

        if (tools_json && tools_json[0] != '\0') {
            inputs.tools = nlohmann::ordered_json::parse(tools_json);
        } else {
            inputs.tools = nlohmann::ordered_json::array();
        }

        common_chat_params chat_params =
            autoparser::peg_generator::generate_parser(tmpl, inputs, parser);

        common_chat_parser_params parser_params(chat_params);
        parser_params.parser.load(chat_params.parser);

        common_chat_msg parsed = common_chat_parse(input, is_partial != 0, parser_params);

        auto * handle = new llama_rs_parsed_chat{};
        handle->message = std::move(parsed);

        *out_handle = handle;

        return LLAMA_RS_PARSE_CHAT_MESSAGE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));
        if (!*out_error) {
            return LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSE_CHAT_MESSAGE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));
        if (!*out_error) {
            return LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSE_CHAT_MESSAGE_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_parsed_chat_free_status llama_rs_parsed_chat_free(
    llama_rs_parsed_chat_handle handle,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    try {
        delete handle;
        return LLAMA_RS_PARSED_CHAT_FREE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_parsed_chat_tool_call_count_status llama_rs_parsed_chat_tool_call_count(
    llama_rs_parsed_chat_handle handle,
    size_t * out_count,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_count) {
        *out_count = 0;
    }
    if (!handle) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_HANDLE_ARG;
    }
    if (!out_count) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_OUT_COUNT_ARG;
    }
    try {
        *out_count = handle->message.tool_calls.size();
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_parsed_chat_tool_call_id_status llama_rs_parsed_chat_tool_call_id(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_string) {
        *out_string = nullptr;
    }
    if (!handle) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_HANDLE_ARG;
    }
    if (!out_string) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_OUT_STRING_ARG;
    }
    try {
        if (index >= handle->message.tool_calls.size()) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_INDEX_OUT_OF_BOUNDS;
        }
        bool alloc_failed = false;
        *out_string = dup_or_set_alloc_flag(handle->message.tool_calls[index].id, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_parsed_chat_tool_call_name_status llama_rs_parsed_chat_tool_call_name(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_string) {
        *out_string = nullptr;
    }
    if (!handle) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_HANDLE_ARG;
    }
    if (!out_string) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_OUT_STRING_ARG;
    }
    try {
        if (index >= handle->message.tool_calls.size()) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_INDEX_OUT_OF_BOUNDS;
        }
        bool alloc_failed = false;
        *out_string = dup_or_set_alloc_flag(handle->message.tool_calls[index].name, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_parsed_chat_tool_call_arguments_status llama_rs_parsed_chat_tool_call_arguments(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_string) {
        *out_string = nullptr;
    }
    if (!handle) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_HANDLE_ARG;
    }
    if (!out_string) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_OUT_STRING_ARG;
    }
    try {
        if (index >= handle->message.tool_calls.size()) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_INDEX_OUT_OF_BOUNDS;
        }
        bool alloc_failed = false;
        *out_string = dup_or_set_alloc_flag(
            handle->message.tool_calls[index].arguments, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_parsed_chat_content_status llama_rs_parsed_chat_content(
    llama_rs_parsed_chat_handle handle,
    char ** out_string,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_string) {
        *out_string = nullptr;
    }
    if (!handle) {
        return LLAMA_RS_PARSED_CHAT_CONTENT_NULL_HANDLE_ARG;
    }
    if (!out_string) {
        return LLAMA_RS_PARSED_CHAT_CONTENT_NULL_OUT_STRING_ARG;
    }
    try {
        bool alloc_failed = false;
        *out_string = dup_or_set_alloc_flag(handle->message.content, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_CONTENT_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_CONTENT_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_CONTENT_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" llama_rs_parsed_chat_reasoning_content_status llama_rs_parsed_chat_reasoning_content(
    llama_rs_parsed_chat_handle handle,
    char ** out_string,
    char ** out_error) {
    if (out_error) {
        *out_error = nullptr;
    }
    if (out_string) {
        *out_string = nullptr;
    }
    if (!handle) {
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_HANDLE_ARG;
    }
    if (!out_string) {
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_OUT_STRING_ARG;
    }
    try {
        bool alloc_failed = false;
        *out_string = dup_or_set_alloc_flag(handle->message.reasoning_content, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error) {
            *out_error = llama_rs_dup_string(err.what());
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (!*out_error) {
                return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_VENDORED_THREW_CXX_EXCEPTION;
    }
}
