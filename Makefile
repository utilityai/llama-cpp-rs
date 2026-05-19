TEST_DEVICE ?=
QWEN_CAPABLE_FEATURES = multimodal_capable,mrope_model

DEVICE_FEATURE = $(if $(TEST_DEVICE),--features $(TEST_DEVICE),)
LLM_BASE_FEATURE_FLAGS = $(DEVICE_FEATURE)
LLM_QWEN_CAPABLE_FEATURE_FLAGS = $(DEVICE_FEATURE) --features $(QWEN_CAPABLE_FEATURES)

CARGO_TEST_LLM_FLAGS = --release --no-fail-fast -p llama-cpp-bindings-tests $(LLM_BASE_FEATURE_FLAGS) -- --test-threads=1
CARGO_TEST_LLM_FLAGS_QWEN_CAPABLE = --release --no-fail-fast -p llama-cpp-bindings-tests $(LLM_QWEN_CAPABLE_FEATURE_FLAGS) -- --test-threads=1


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

node_modules: package-lock.json
	npm ci
	touch node_modules

package-lock.json: package.json
	npm install --package-lock-only

.PHONY: clean.cmake
clean.cmake:
	rm -rf target/llama-cpp-cmake-build

.PHONY: clippy
clippy: clippy.core clippy.tests.base clippy.tests.qwen_capable

.PHONY: clippy.core
clippy.core:
	cargo clippy --all-targets -p llama-cpp-log-decoder -- -D warnings
	cargo clippy --all-targets -p llama-cpp-bindings $(DEVICE_FEATURE) -- -D warnings

.PHONY: clippy.tests.base
clippy.tests.base:
	cargo clippy --all-targets -p llama-cpp-bindings-tests $(LLM_BASE_FEATURE_FLAGS) -- -D warnings

.PHONY: clippy.tests.qwen_capable
clippy.tests.qwen_capable:
	cargo clippy --all-targets -p llama-cpp-bindings-tests $(LLM_QWEN_CAPABLE_FEATURE_FLAGS) -- -D warnings

.PHONY: coverage
coverage: node_modules
	cargo llvm-cov clean --workspace
	cargo llvm-cov --no-report -p llama-cpp-log-decoder
	cargo llvm-cov --no-report -p llama-cpp-bindings-types
	cargo llvm-cov --no-report -p llama-cpp-bindings --lib $(DEVICE_FEATURE)
	$(DEEPSEEK_R1_DISTILL_LLAMA_8B_ENV) cargo llvm-cov --no-report --no-fail-fast -p llama-cpp-bindings-tests $(LLM_BASE_FEATURE_FLAGS) -- --test-threads=1
	$(GLM4_7_FLASH_ENV) cargo llvm-cov --no-report --no-fail-fast -p llama-cpp-bindings-tests $(LLM_BASE_FEATURE_FLAGS) -- --test-threads=1
	$(QWEN3_5_0_8B_ENV) cargo llvm-cov --no-report --no-fail-fast -p llama-cpp-bindings-tests $(LLM_QWEN_CAPABLE_FEATURE_FLAGS) -- --test-threads=1
	$(QWEN3_6_35B_A3B_ENV) cargo llvm-cov --no-report --no-fail-fast -p llama-cpp-bindings-tests $(LLM_QWEN_CAPABLE_FEATURE_FLAGS) -- --test-threads=1
	cargo llvm-cov report --json --output-path target/llvm-cov.json
	cargo llvm-cov report --lcov --output-path target/lcov.info
	cargo llvm-cov report
	npx rust-coverage-check target/llvm-cov.json \
		--workspace-root $(CURDIR) \
		--gated llama-cpp-bindings=95 \
		--gated llama-cpp-log-decoder=99 \
		--gated llama-cpp-bindings-types=99

.PHONY: coverage-clean
coverage-clean:
	cargo llvm-cov clean --workspace
	rm -rf target/llvm-cov-target
	rm -f target/llvm-cov.json target/lcov.info

.PHONY: coverage-report
coverage-report:
	cargo llvm-cov report --html

.PHONY: fmt
fmt:
	cargo fmt --all

.PHONY: fmt.check
fmt.check:
	cargo fmt --all --check

.PHONY: test
test: test.unit test.llms

.PHONY: test.deepseek_r1_distill_llama_8b
test.deepseek_r1_distill_llama_8b: clippy.core clippy.tests.base
	$(DEEPSEEK_R1_DISTILL_LLAMA_8B_ENV) cargo test $(CARGO_TEST_LLM_FLAGS)

.PHONY: test.glm4_7_flash
test.glm4_7_flash: clippy.core clippy.tests.base
	$(GLM4_7_FLASH_ENV) cargo test $(CARGO_TEST_LLM_FLAGS)

.PHONY: test.llms
test.llms: \
	test.deepseek_r1_distill_llama_8b \
	test.glm4_7_flash \
	test.qwen3.5_0.8B \
	test.qwen3.6_35b_a3b

.PHONY: test.qwen3.5_0.8B
test.qwen3.5_0.8B: clippy.core clippy.tests.qwen_capable
	$(QWEN3_5_0_8B_ENV) cargo test $(CARGO_TEST_LLM_FLAGS_QWEN_CAPABLE)

.PHONY: test.qwen3.6_35b_a3b
test.qwen3.6_35b_a3b: clippy.core clippy.tests.qwen_capable
	$(QWEN3_6_35B_A3B_ENV) cargo test $(CARGO_TEST_LLM_FLAGS_QWEN_CAPABLE)

.PHONY: test.unit
test.unit: clippy.core
	cargo test -p llama-cpp-log-decoder
	cargo test -p llama-cpp-bindings $(DEVICE_FEATURE)
