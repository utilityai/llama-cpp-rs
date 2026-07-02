#include "wrapper_tool_calls.h"
#include <nlohmann/json.hpp> // IWYU pragma: keep
#include <nlohmann/json_fwd.hpp>
#include "wrapper_token_text.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat-auto-parser-helpers.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <exception>
#include <memory>
#include <new>
#include <string>

using wrapper_helpers::token_text_or_empty;

namespace {

// Render the chat template with a deterministic tool-call assistant turn and
// diff it against the no-tool-call variant. Returns the raw section between
// the model's tool-call open/close markers — i.e. the `<...>{...}</...>`
// fragment the model is expected to emit, with any reasoning prelude removed.
//
// We deliberately reproduce the autoparser's diff-based approach (so the
// detected markers come from the model's actual template behavior, not from a
// hardcoded list), but use plain-ASCII synthetic names where the upstream
// autoparser uses sentinel strings that some Jinja templates choke on.
auto detect_tool_call_haystack(
    const common_chat_template & tmpl,
    const autoparser::analyze_reasoning & reasoning) -> std::string {
    nlohmann::ordered_json const user_msg = {
        { "role",    "user"                },
        { "content", "Please use the tool" }
    };
    nlohmann::ordered_json const assistant_no_tools = {
        { "role",    "assistant"      },
        { "content", "Sure, calling." }
    };
    nlohmann::ordered_json const first_tool_call = {
        { "id",       "call_001"  },
        { "type",     "function"  },
        { "function", {
            { "name",      "tool_first" },
            { "arguments", {
                { "arg_first", "XXXX" },
                { "arg_second", "YYYY" },
            }}
        }}
    };
    nlohmann::ordered_json const assistant_with_tools = {
        { "role",       "assistant"                                                  },
        { "content",    ""                                                           },
        { "tool_calls", nlohmann::ordered_json::array({ first_tool_call })           }
    };
    nlohmann::ordered_json const tool_definition = {
        { "type",     "function"  },
        { "function", {
            { "name",        "tool_first"           },
            { "description", "First test tool"      },
            { "parameters", {
                { "type", "object" },
                { "properties", {
                    { "arg_first", { { "type", "string" }, { "description", "first arg" } } },
                    { "arg_second", { { "type", "string" }, { "description", "second arg" } } },
                }},
                { "required", nlohmann::ordered_json::array({ "arg_first", "arg_second" }) },
            }}
        }}
    };

    template_params params_no_tools;
    params_no_tools.messages              = nlohmann::ordered_json::array({ user_msg, assistant_no_tools });
    params_no_tools.tools                 = nlohmann::ordered_json::array({ tool_definition });
    params_no_tools.add_generation_prompt = false;
    params_no_tools.enable_thinking       = true;

    template_params params_with_tools = params_no_tools;
    params_with_tools.messages =
        nlohmann::ordered_json::array({ user_msg, assistant_with_tools });

    std::string const output_no_tools = autoparser::apply_template(tmpl, params_no_tools);
    std::string const output_with_tools = autoparser::apply_template(tmpl, params_with_tools);

    if (output_no_tools.empty() || output_with_tools.empty()) {
        return {};
    }

    diff_split const diff = calculate_diff_split(output_no_tools, output_with_tools);
    std::string haystack = diff.right;

    // Strip reasoning markers so the surrounding tool-call markers can be
    // located reliably — the autoparser does the same for the JSON-native
    // path.
    auto remove_first = [&haystack](const std::string & needle) -> void {
        if (needle.empty()) {
            return;
        }
        auto pos = haystack.find(needle);
        if (pos != std::string::npos) {
            haystack.erase(pos, needle.length());
        }
    };

    remove_first(reasoning.start);
    remove_first(reasoning.end);

    return haystack;
}

}  // namespace

extern "C" auto llama_rs_compute_tool_call_haystack(
    const struct llama_model * model,
    char ** out_haystack,
    char ** out_error) -> llama_rs_compute_tool_call_haystack_status {
    if (out_haystack != nullptr) {
        *out_haystack = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_MODEL_ARG;
    }
    if (out_haystack == nullptr) {
        return LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_OUT_HAYSTACK_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_NULL_OUT_ERROR_ARG;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr) {
            return LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_OK;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            return LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_OK;
        }

        std::string const bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string const eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template const tmpl(tmpl_src, bos_token, eos_token);
        auto jinja_caps = tmpl.original_caps();
        autoparser::analyze_reasoning const reasoning(tmpl, jinja_caps.supports_tool_calls);

        std::string const haystack = detect_tool_call_haystack(tmpl, reasoning);
        if (haystack.empty()) {
            return LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_OK;
        }

        char * haystack_dup = llama_rs_dup_string(haystack);
        if (haystack_dup == nullptr) {
            return LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_ERROR_STRING_ALLOCATION_FAILED;
        }

        *out_haystack = haystack_dup;

        return LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_COMPUTE_TOOL_CALL_HAYSTACK_THREW_CXX_EXCEPTION);
    }
}

extern "C" auto llama_rs_diagnose_tool_call_synthetic_renders(
    const struct llama_model * model,
    char ** out_no_tools,
    char ** out_with_tools,
    char ** out_error) -> llama_rs_diagnose_tool_call_synthetic_renders_status {
    if (out_no_tools != nullptr) {
        *out_no_tools = nullptr;
    }
    if (out_with_tools != nullptr) {
        *out_with_tools = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_MODEL_ARG;
    }
    if (out_no_tools == nullptr) {
        return LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_NO_TOOLS_ARG;
    }
    if (out_with_tools == nullptr) {
        return LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_WITH_TOOLS_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_NULL_OUT_ERROR_ARG;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr) {
            return LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_OK;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            return LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_OK;
        }

        std::string const bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string const eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template const tmpl(tmpl_src, bos_token, eos_token);

        nlohmann::ordered_json const user_msg = {
            { "role",    "user"                },
            { "content", "Please use the tool" }
        };
        nlohmann::ordered_json const assistant_no_tools = {
            { "role",    "assistant"      },
            { "content", "Sure, calling." }
        };
        nlohmann::ordered_json const first_tool_call = {
            { "id",       "call_001"  },
            { "type",     "function"  },
            { "function", {
                { "name",      "tool_first" },
                { "arguments", {
                    { "arg_first", "XXXX" },
                    { "arg_second", "YYYY" },
                }}
            }}
        };
        nlohmann::ordered_json const assistant_with_tools = {
            { "role",       "assistant"                                                  },
            { "content",    ""                                                           },
            { "tool_calls", nlohmann::ordered_json::array({ first_tool_call })           }
        };
        nlohmann::ordered_json const tool_definition = {
            { "type",     "function"  },
            { "function", {
                { "name",        "tool_first"           },
                { "description", "First test tool"      },
                { "parameters", {
                    { "type", "object" },
                    { "properties", {
                        { "arg_first", { { "type", "string" }, { "description", "first arg" } } },
                        { "arg_second", { { "type", "string" }, { "description", "second arg" } } },
                    }},
                    { "required", nlohmann::ordered_json::array({ "arg_first", "arg_second" }) },
                }}
            }}
        };

        template_params params_no_tools;
        params_no_tools.messages              = nlohmann::ordered_json::array({ user_msg, assistant_no_tools });
        params_no_tools.tools                 = nlohmann::ordered_json::array({ tool_definition });
        params_no_tools.add_generation_prompt = false;
        params_no_tools.enable_thinking       = true;

        template_params params_with_tools = params_no_tools;
        params_with_tools.messages =
            nlohmann::ordered_json::array({ user_msg, assistant_with_tools });

        std::string const output_a = autoparser::apply_template(tmpl, params_no_tools);
        std::string const output_b = autoparser::apply_template(tmpl, params_with_tools);

        std::unique_ptr<char[]> a_dup(llama_rs_dup_string(output_a));
        std::unique_ptr<char[]> b_dup(llama_rs_dup_string(output_b));

        if ((a_dup == nullptr) || (b_dup == nullptr)) {
            return LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_ERROR_STRING_ALLOCATION_FAILED;
        }

        *out_no_tools = a_dup.release();
        *out_with_tools = b_dup.release();

        return LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            out_error,
            LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_ERROR_STRING_ALLOCATION_FAILED,
            LLAMA_RS_DIAGNOSE_TOOL_CALL_SYNTHETIC_RENDERS_THREW_CXX_EXCEPTION);
    }
}
