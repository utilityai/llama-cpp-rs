FEATURES = sampler,llguidance
CARGO_TEST_LLM_FLAGS = --lib -p llama-cpp-bindings --features tests_that_use_llms,$(FEATURES) -- --test-threads=1
CARGO_COV_LLM_FLAGS = --lib --features tests_that_use_llms,$(FEATURES) -p llama-cpp-bindings

QWEN3_5_0_8B_ENV = \
	LLAMA_TEST_HF_REPO=unsloth/Qwen3.5-0.8B-GGUF \
	LLAMA_TEST_HF_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
	LLAMA_TEST_HF_MMPROJ=mmproj-F16.gguf \
	LLAMA_TEST_HF_EMBED_REPO=Qwen/Qwen3-Embedding-0.6B-GGUF \
	LLAMA_TEST_HF_EMBED_MODEL=Qwen3-Embedding-0.6B-Q8_0.gguf \
	LLAMA_TEST_HF_ENCODER_REPO=Xiaojian9992024/t5-small-GGUF \
	LLAMA_TEST_HF_ENCODER_MODEL=t5-small.bf16.gguf

.PHONY: test.unit
test.unit: clippy
	cargo test --lib -p llama-cpp-bindings --features $(FEATURES)

.PHONY: test.qwen3.5_0.8B
test.qwen3.5_0.8B: clippy
	$(QWEN3_5_0_8B_ENV) cargo test $(CARGO_TEST_LLM_FLAGS)

.PHONY: test.qwen3.5_0.8B.coverage.run
test.qwen3.5_0.8B.coverage.run: clippy
	$(QWEN3_5_0_8B_ENV) cargo llvm-cov $(CARGO_COV_LLM_FLAGS) -- --test-threads=1

.PHONY: test.qwen3.5_0.8B.coverage

test.qwen3.5_0.8B.coverage: clippy
	$(QWEN3_5_0_8B_ENV) cargo llvm-cov $(CARGO_COV_LLM_FLAGS) --fail-under-lines 99.5 -- --test-threads=1

.PHONY: test.qwen3.5_0.8B.coverage.json
test.qwen3.5_0.8B.coverage.json: test.qwen3.5_0.8B.coverage.run
	cargo llvm-cov report -p llama-cpp-bindings --json --output-path target/coverage.json

.PHONY: test.qwen3.5_0.8B.coverage.html
test.qwen3.5_0.8B.coverage.html: test.qwen3.5_0.8B.coverage.run
	cargo llvm-cov report -p llama-cpp-bindings --html

.PHONY: test.llms
test.llms: test.qwen3.5_0.8B

.PHONY: test
test: test.unit test.llms

.PHONY: fmt
fmt:
	cargo fmt --all --check

.PHONY: clippy
clippy:
	cargo clippy --all-targets -p llama-cpp-bindings --features $(FEATURES) -- -D warnings

.PHONY: clean.cmake
clean.cmake:
	rm -rf target/llama-cpp-cmake-build
