#include "chunked_thinking.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat.h"

#include <algorithm>
#include <exception>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>

namespace marker_probes {

namespace {

constexpr std::string_view REASON_PROBE   = "__PADDLER_REASON_PROBE_3F4A8C__";
constexpr std::string_view RESPONSE_PROBE = "__PADDLER_RESPONSE_PROBE_3F4A8C__";

std::string trim_copy(std::string_view input) {
    auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) {
        return {};
    }
    auto last = input.find_last_not_of(" \t\r\n");
    return std::string(input.substr(first, last - first + 1));
}

bool render_template(const common_chat_template & tmpl,
                     const autoparser::generation_params & params,
                     std::string & out) {
    try {
        out = common_chat_template_direct_apply(tmpl, params);
        return true;
    } catch (const std::exception &) {
        return false;
    } catch (...) {
        return false;
    }
}

autoparser::generation_params plain_text_params() {
    autoparser::generation_params params;
    params.add_generation_prompt = false;
    params.enable_thinking = true;
    params.is_inference = false;
    params.add_inference = false;
    params.mark_input = false;
    params.messages = nlohmann::ordered_json::array({
        nlohmann::ordered_json{ { "role", "user" }, { "content", "U" } },
        nlohmann::ordered_json{ { "role", "assistant" }, { "content", std::string(RESPONSE_PROBE) } },
    });
    return params;
}

autoparser::generation_params chunked_thinking_params() {
    autoparser::generation_params params;
    params.add_generation_prompt = false;
    params.enable_thinking = true;
    params.is_inference = false;
    params.add_inference = false;
    params.mark_input = false;
    params.messages = nlohmann::ordered_json::array({
        nlohmann::ordered_json{ { "role", "user" }, { "content", "U" } },
        nlohmann::ordered_json{
            { "role", "assistant" },
            { "content", nlohmann::ordered_json::array({
                  nlohmann::ordered_json{ { "type", "thinking" }, { "thinking", std::string(REASON_PROBE) } },
                  nlohmann::ordered_json{ { "type", "text" }, { "text", std::string(RESPONSE_PROBE) } },
              }) },
        },
    });
    return params;
}

bool contains(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

}  // namespace

probe_result chunked_thinking(const common_chat_template & tmpl) {
    probe_result result;

    std::string render_plain;
    if (!render_template(tmpl, plain_text_params(), render_plain)) {
        return result;
    }

    std::string render_chunked;
    if (!render_template(tmpl, chunked_thinking_params(), render_chunked)) {
        return result;
    }

    if (!contains(render_chunked, REASON_PROBE) || !contains(render_chunked, RESPONSE_PROBE)) {
        return result;
    }

    const std::size_t plain_size = render_plain.size();
    const std::size_t chunked_size = render_chunked.size();
    const std::size_t min_size = std::min(plain_size, chunked_size);

    std::size_t common_prefix = 0;
    while (common_prefix < min_size && render_plain[common_prefix] == render_chunked[common_prefix]) {
        ++common_prefix;
    }

    std::size_t common_suffix = 0;
    while (common_suffix < min_size - common_prefix
           && render_plain[plain_size - 1 - common_suffix] == render_chunked[chunked_size - 1 - common_suffix]) {
        ++common_suffix;
    }

    if (common_prefix + common_suffix > chunked_size) {
        return result;
    }

    std::string_view diff_slice(render_chunked);
    diff_slice = diff_slice.substr(common_prefix, chunked_size - common_prefix - common_suffix);

    auto reason_pos = diff_slice.find(REASON_PROBE);
    if (reason_pos == std::string_view::npos) {
        return result;
    }

    std::string start = trim_copy(diff_slice.substr(0, reason_pos));
    std::string end = trim_copy(diff_slice.substr(reason_pos + REASON_PROBE.size()));

    if (start.empty() || end.empty()) {
        return result;
    }
    if (contains(start, REASON_PROBE) || contains(start, RESPONSE_PROBE)) {
        return result;
    }
    if (contains(end, REASON_PROBE) || contains(end, RESPONSE_PROBE)) {
        return result;
    }

    result.start = std::move(start);
    result.end = std::move(end);
    result.found = true;
    return result;
}

}  // namespace marker_probes
