#include "marker_probe.h"

#include "chunked_thinking.h"

#include <vector>

namespace marker_probes {

const std::vector<probe_fn> & registered() {
    static const std::vector<probe_fn> probes = {
        chunked_thinking,
    };
    return probes;
}

}  // namespace marker_probes
