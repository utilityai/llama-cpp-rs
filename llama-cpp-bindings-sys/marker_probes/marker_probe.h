#pragma once

#include "llama.cpp/common/chat.h"

#include <string>
#include <vector>

namespace marker_probes {

struct probe_result {
    std::string start;
    std::string end;
    bool found = false;
};

using probe_fn = probe_result (*)(const common_chat_template &);

const std::vector<probe_fn> & registered();

}  // namespace marker_probes
