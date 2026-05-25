TEST_DEVICE ?=

DEVICE_FEATURE = $(if $(TEST_DEVICE),--features $(TEST_DEVICE),)

node_modules: package-lock.json
	npm ci
	touch node_modules

package-lock.json: package.json
	npm install --package-lock-only

.PHONY: clean.cmake
clean.cmake:
	rm -rf target/llama-cpp-cmake-build

.PHONY: clippy
clippy:
	cargo clippy --workspace --all-targets $(DEVICE_FEATURE) -- -D warnings

.PHONY: coverage
coverage: node_modules
	cargo llvm-cov clean --workspace
	cargo llvm-cov --no-report --no-fail-fast --workspace $(DEVICE_FEATURE)
	cargo llvm-cov report --json --output-path target/llvm-cov.json
	cargo llvm-cov report --lcov --output-path target/lcov.info
	cargo llvm-cov report
	npx rust-coverage-check target/llvm-cov.json \
		--workspace-root $(CURDIR) \
		--gated llama-cpp-bindings=95 \
		--gated llama-cpp-log-decoder=100 \
		--gated llama-cpp-bindings-types=100 \
		--gated llama-cpp-test-harness=99 \
		--gated llama-cpp-test-harness-macros=100

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

.PHONY: test.harness
test.harness: clippy
	cargo test -p llama-cpp-test-harness-macros -p llama-cpp-test-harness $(DEVICE_FEATURE)

.PHONY: test.llms
test.llms: clippy test.harness test.unit
	cargo test --no-fail-fast -p llama-cpp-bindings-tests $(DEVICE_FEATURE)

.PHONY: test.unit
test.unit: clippy
	cargo test -p llama-cpp-log-decoder -p llama-cpp-bindings $(DEVICE_FEATURE)
