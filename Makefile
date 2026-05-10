FEATURES = sampler
TEST_FEATURES =
QWEN_CAPABLE_FEATURES = multimodal_capable,mrope_model
CARGO_TEST_LLM_FLAGS = --no-fail-fast -p llama-cpp-bindings-tests $(if $(TEST_FEATURES),--features $(TEST_FEATURES),) -- --test-threads=1
CARGO_TEST_LLM_FLAGS_QWEN_CAPABLE = --no-fail-fast -p llama-cpp-bindings-tests $(if $(TEST_FEATURES),--features $(TEST_FEATURES),) --features $(QWEN_CAPABLE_FEATURES) -- --test-threads=1
CARGO_COV_LLM_FLAGS = -p llama-cpp-bindings-tests $(if $(TEST_FEATURES),--features $(TEST_FEATURES),) --features $(QWEN_CAPABLE_FEATURES)

QWEN3_5_0_8B_ENV = \
	LLAMA_TEST_HF_REPO=unsloth/Qwen3.5-0.8B-GGUF \
	LLAMA_TEST_HF_MODEL=Qwen3.5-0.8B-Q4_K_M.gguf \
	LLAMA_TEST_HF_MMPROJ=mmproj-F16.gguf \
	LLAMA_TEST_HF_EMBED_REPO=Qwen/Qwen3-Embedding-0.6B-GGUF \
	LLAMA_TEST_HF_EMBED_MODEL=Qwen3-Embedding-0.6B-Q8_0.gguf \
	LLAMA_TEST_HF_ENCODER_REPO=Xiaojian9992024/t5-small-GGUF \
	LLAMA_TEST_HF_ENCODER_MODEL=t5-small.bf16.gguf

QWEN3_6_35B_A3B_ENV = \
	LLAMA_TEST_HF_REPO=unsloth/Qwen3.6-35B-A3B-GGUF \
	LLAMA_TEST_HF_MODEL=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
	LLAMA_TEST_HF_MMPROJ=mmproj-F16.gguf \
	LLAMA_TEST_HF_EMBED_REPO=Qwen/Qwen3-Embedding-0.6B-GGUF \
	LLAMA_TEST_HF_EMBED_MODEL=Qwen3-Embedding-0.6B-Q8_0.gguf \
	LLAMA_TEST_HF_ENCODER_REPO=Xiaojian9992024/t5-small-GGUF \
	LLAMA_TEST_HF_ENCODER_MODEL=t5-small.bf16.gguf

GLM4_7_FLASH_ENV = \
	LLAMA_TEST_HF_REPO=unsloth/GLM-4.7-Flash-GGUF \
	LLAMA_TEST_HF_MODEL=GLM-4.7-Flash-Q4_K_M.gguf \
	LLAMA_TEST_HF_EMBED_REPO=Qwen/Qwen3-Embedding-0.6B-GGUF \
	LLAMA_TEST_HF_EMBED_MODEL=Qwen3-Embedding-0.6B-Q8_0.gguf \
	LLAMA_TEST_HF_ENCODER_REPO=Xiaojian9992024/t5-small-GGUF \
	LLAMA_TEST_HF_ENCODER_MODEL=t5-small.bf16.gguf

DEEPSEEK_R1_DISTILL_LLAMA_8B_ENV = \
	LLAMA_TEST_HF_REPO=unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF \
	LLAMA_TEST_HF_MODEL=DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf \
	LLAMA_TEST_HF_EMBED_REPO=Qwen/Qwen3-Embedding-0.6B-GGUF \
	LLAMA_TEST_HF_EMBED_MODEL=Qwen3-Embedding-0.6B-Q8_0.gguf \
	LLAMA_TEST_HF_ENCODER_REPO=Xiaojian9992024/t5-small-GGUF \
	LLAMA_TEST_HF_ENCODER_MODEL=t5-small.bf16.gguf

.PHONY: test.unit
test.unit: clippy
	cargo test -p llama-cpp-bindings --features $(FEATURES)

.PHONY: test.qwen3.5_0.8B
test.qwen3.5_0.8B: clippy
	$(QWEN3_5_0_8B_ENV) cargo test $(CARGO_TEST_LLM_FLAGS_QWEN_CAPABLE)

.PHONY: test.qwen3.6_35b_a3b
test.qwen3.6_35b_a3b: clippy
	$(QWEN3_6_35B_A3B_ENV) cargo test $(CARGO_TEST_LLM_FLAGS_QWEN_CAPABLE)

.PHONY: test.glm4_7_flash
test.glm4_7_flash: clippy
	$(GLM4_7_FLASH_ENV) cargo test $(CARGO_TEST_LLM_FLAGS)

.PHONY: test.deepseek_r1_distill_llama_8b
test.deepseek_r1_distill_llama_8b: clippy
	$(DEEPSEEK_R1_DISTILL_LLAMA_8B_ENV) cargo test $(CARGO_TEST_LLM_FLAGS)

.PHONY: test.qwen3.5_0.8B.coverage.run
test.qwen3.5_0.8B.coverage.run: clippy
	cargo llvm-cov clean --workspace
	cargo llvm-cov --no-report -p llama-cpp-bindings --features $(FEATURES) --lib
	$(QWEN3_5_0_8B_ENV) cargo llvm-cov --no-report $(CARGO_COV_LLM_FLAGS) -- --test-threads=1

.PHONY: test.qwen3.5_0.8B.coverage
test.qwen3.5_0.8B.coverage: test.qwen3.5_0.8B.coverage.run
	cargo llvm-cov report -p llama-cpp-bindings --fail-under-lines 98.5

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
	cargo clippy --all-targets -p llama-cpp-bindings-tests $(if $(TEST_FEATURES),--features $(TEST_FEATURES),) -- -D warnings

.PHONY: clean.cmake
clean.cmake:
	rm -rf target/llama-cpp-cmake-build
