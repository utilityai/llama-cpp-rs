---
name: run-all-tests
description: Runs every test suite in the workspace on the fastest available device. Use when the user asks to run the tests, run all the tests, run the full test suite, or check that everything still passes.
---

# Running all tests

Run every test suite in the workspace, picking the fastest compiled device backend for the host. 

## Step 1: detect the device

Run this once at the start and echo the chosen device:

```bash
if [[ "$OSTYPE" == "darwin"* ]]; then
  DEVICE=metal
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DEVICE=cuda
else
  DEVICE=cpu
fi
echo "Device: $DEVICE"
```

`$DEVICE` selects the backend feature for every suite in Step 2, including `test.unit`. Passing the same device through every target keeps the cmake hash stable, so llama.cpp is compiled once and reused across all suites.

## Step 2: run the suites

Sequentially, from the workspace root. 

Copy this checklist and tick each item as the suite completes:

```
Test progress:
- [ ] make test.unit
- [ ] make test.qwen3.5_0.8B
- [ ] make test.qwen3.6_35b_a3b
- [ ] make test.glm4_7_flash
- [ ] make test.deepseek_r1_distill_llama_8b
```

Translate `$DEVICE` into the value the Makefile expects. `TEST_DEVICE` holds **only** the backend name (`cuda` / `metal` / `vulkan` / `rocm`), or empty for CPU since there is no `cpu` feature:

```bash
[ "$DEVICE" = "cpu" ] && FEAT= || FEAT="$DEVICE"
```

Then run exactly:

```bash
make test.unit TEST_DEVICE="$FEAT"
make test.qwen3.5_0.8B TEST_DEVICE="$FEAT"
make test.qwen3.6_35b_a3b TEST_DEVICE="$FEAT"
make test.glm4_7_flash TEST_DEVICE="$FEAT"
make test.deepseek_r1_distill_llama_8b TEST_DEVICE="$FEAT"
```

The Makefile's `$(if $(TEST_DEVICE),--features $(TEST_DEVICE),)` already skips the `--features` flag when `$FEAT` is empty, so the CPU path needs no further special-casing.

Do not run `make test.llms` or `make test`. Those bundle every LLM suite into one cargo invocation, which loses per-suite failure attribution and breaks the checklist above.

## Step 3: rules during the run

- **Serialize GPU suites.** When `$DEVICE` is `cuda` or `metal`, run test suites sequentially to avoid device contention.
- **Per-test 30 s budget.** Flag any individual test that exceeds 30 s wall-clock. That is a real bug — production or test — not flakiness.

## Step 4: report

After all suites finish, sum up the results in an actionable report.

