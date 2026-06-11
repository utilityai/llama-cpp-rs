#include "wrapper_chat_parse.h"
#include <nlohmann/json.hpp> // IWYU pragma: keep
#include <nlohmann/json_fwd.hpp>
#include "peg-parser.h"
#include "wrapper_token_text.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>

using wrapper_helpers::token_text_or_empty;

struct llama_rs_parsed_chat {
    common_chat_msg message;
};

struct llama_rs_chat_parser {
    autoparser::autoparser parser;
};

namespace {
void dup_or_set_alloc_flag(const std::string & source, char ** out_dup, bool * out_alloc_failed) {
    *out_dup = llama_rs_dup_string(source);
    *out_alloc_failed = (*out_dup == nullptr);
}
} // namespace

extern "C" auto llama_rs_chat_parser_create(
    const struct llama_model * model,
    const char * reasoning_open,
    const char * reasoning_close,
    llama_rs_chat_parser_handle * out_parser,
    char ** out_error) -> llama_rs_chat_parser_create_status {
    if (out_parser != nullptr) {
        *out_parser = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_CHAT_PARSER_CREATE_NULL_MODEL_ARG;
    }
    if (out_parser == nullptr) {
        return LLAMA_RS_CHAT_PARSER_CREATE_NULL_OUT_PARSER_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CHAT_PARSER_CREATE_NULL_OUT_ERROR_ARG;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr) {
            return LLAMA_RS_CHAT_PARSER_CREATE_MODEL_HAS_NO_CHAT_TEMPLATE;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            return LLAMA_RS_CHAT_PARSER_CREATE_MODEL_HAS_NO_VOCAB;
        }

        std::string const bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string const eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template const tmpl(tmpl_src, bos_token, eos_token);

        auto parser_handle = std::make_unique<llama_rs_chat_parser>();
        parser_handle->parser.analyze_template(tmpl);

        if (parser_handle->parser.reasoning.mode == autoparser::reasoning_mode::NONE
            && reasoning_open != nullptr && reasoning_close != nullptr
            && *reasoning_open != '\0' && *reasoning_close != '\0') {
            parser_handle->parser.reasoning.mode  = autoparser::reasoning_mode::TAG_BASED;
            parser_handle->parser.reasoning.start = reasoning_open;
            parser_handle->parser.reasoning.end   = reasoning_close;
        }

        *out_parser = parser_handle.release();

        return LLAMA_RS_CHAT_PARSER_CREATE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_CHAT_PARSER_CREATE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));
        if (*out_error == nullptr) {
            return LLAMA_RS_CHAT_PARSER_CREATE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_CHAT_PARSER_CREATE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));
        if (*out_error == nullptr) {
            return LLAMA_RS_CHAT_PARSER_CREATE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_CHAT_PARSER_CREATE_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_chat_parser_free(
    llama_rs_chat_parser_handle parser,
    char ** out_error) -> llama_rs_chat_parser_free_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    try {
        const std::unique_ptr<llama_rs_chat_parser> reclaimed(parser);
        return LLAMA_RS_CHAT_PARSER_FREE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_CHAT_PARSER_FREE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_CHAT_PARSER_FREE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_CHAT_PARSER_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_CHAT_PARSER_FREE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_CHAT_PARSER_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_parse_chat_message(
    llama_rs_chat_parser_handle parser,
    const char * tools_json,
    const char * input,
    int is_partial,
    llama_rs_parsed_chat_handle * out_handle,
    char ** out_error) -> llama_rs_parse_chat_message_status {
    if (out_handle != nullptr) {
        *out_handle = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (parser == nullptr) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_PARSER_ARG;
    }
    if (input == nullptr) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_INPUT_ARG;
    }
    if (out_handle == nullptr) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_HANDLE_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_ERROR_ARG;
    }

    try {
        autoparser::generation_params inputs;

        if ((tools_json != nullptr) && *tools_json != '\0') {
            inputs.tools = nlohmann::ordered_json::parse(tools_json);
        } else {
            inputs.tools = nlohmann::ordered_json::array();
        }

        common_peg_arena const chat_parser = parser->parser.build_parser(inputs, std::string());

        common_chat_parser_params parser_params;
        parser_params.format = COMMON_CHAT_FORMAT_PEG_NATIVE;

        common_chat_msg parsed =
            common_chat_peg_parse(chat_parser, input, is_partial != 0, parser_params);

        auto handle = std::make_unique<llama_rs_parsed_chat>();
        handle->message = std::move(parsed);

        *out_handle = handle.release();

        return LLAMA_RS_PARSE_CHAT_MESSAGE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));
        if (*out_error == nullptr) {
            return LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSE_CHAT_MESSAGE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));
        if (*out_error == nullptr) {
            return LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSE_CHAT_MESSAGE_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_parsed_chat_free(
    llama_rs_parsed_chat_handle handle,
    char ** out_error) -> llama_rs_parsed_chat_free_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    try {
        const std::unique_ptr<llama_rs_parsed_chat> reclaimed(handle);
        return LLAMA_RS_PARSED_CHAT_FREE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_parsed_chat_tool_call_count(
    llama_rs_parsed_chat_handle handle,
    size_t * out_count,
    char ** out_error) -> llama_rs_parsed_chat_tool_call_count_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_count != nullptr) {
        *out_count = 0;
    }
    if (handle == nullptr) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_HANDLE_ARG;
    }
    if (out_count == nullptr) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_OUT_COUNT_ARG;
    }
    try {
        *out_count = handle->message.tool_calls.size();
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_parsed_chat_tool_call_id(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error) -> llama_rs_parsed_chat_tool_call_id_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_string != nullptr) {
        *out_string = nullptr;
    }
    if (handle == nullptr) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_HANDLE_ARG;
    }
    if (out_string == nullptr) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_OUT_STRING_ARG;
    }
    try {
        if (index >= handle->message.tool_calls.size()) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_INDEX_OUT_OF_BOUNDS;
        }
        bool alloc_failed = false;
        dup_or_set_alloc_flag(handle->message.tool_calls[index].id, out_string, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_parsed_chat_tool_call_name(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error) -> llama_rs_parsed_chat_tool_call_name_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_string != nullptr) {
        *out_string = nullptr;
    }
    if (handle == nullptr) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_HANDLE_ARG;
    }
    if (out_string == nullptr) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_OUT_STRING_ARG;
    }
    try {
        if (index >= handle->message.tool_calls.size()) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_INDEX_OUT_OF_BOUNDS;
        }
        bool alloc_failed = false;
        dup_or_set_alloc_flag(handle->message.tool_calls[index].name, out_string, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_parsed_chat_tool_call_arguments(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    char ** out_string,
    char ** out_error) -> llama_rs_parsed_chat_tool_call_arguments_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_string != nullptr) {
        *out_string = nullptr;
    }
    if (handle == nullptr) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_HANDLE_ARG;
    }
    if (out_string == nullptr) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_OUT_STRING_ARG;
    }
    try {
        if (index >= handle->message.tool_calls.size()) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_INDEX_OUT_OF_BOUNDS;
        }
        bool alloc_failed = false;
        dup_or_set_alloc_flag(
            handle->message.tool_calls[index].arguments, out_string, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_parsed_chat_content(
    llama_rs_parsed_chat_handle handle,
    char ** out_string,
    char ** out_error) -> llama_rs_parsed_chat_content_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_string != nullptr) {
        *out_string = nullptr;
    }
    if (handle == nullptr) {
        return LLAMA_RS_PARSED_CHAT_CONTENT_NULL_HANDLE_ARG;
    }
    if (out_string == nullptr) {
        return LLAMA_RS_PARSED_CHAT_CONTENT_NULL_OUT_STRING_ARG;
    }
    try {
        bool alloc_failed = false;
        dup_or_set_alloc_flag(handle->message.content, out_string, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_CONTENT_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_CONTENT_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_CONTENT_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_parsed_chat_reasoning_content(
    llama_rs_parsed_chat_handle handle,
    char ** out_string,
    char ** out_error) -> llama_rs_parsed_chat_reasoning_content_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_string != nullptr) {
        *out_string = nullptr;
    }
    if (handle == nullptr) {
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_HANDLE_ARG;
    }
    if (out_string == nullptr) {
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_OUT_STRING_ARG;
    }
    try {
        bool alloc_failed = false;
        dup_or_set_alloc_flag(handle->message.reasoning_content, out_string, &alloc_failed);
        if (alloc_failed) {
            return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_VENDORED_THREW_CXX_EXCEPTION;
    }
}
