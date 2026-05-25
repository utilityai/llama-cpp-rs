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

Translate `$DEVICE` into the value the Makefile expects. `TEST_DEVICE` holds **only** the backend name (`cuda` / `metal` / `vulkan` / `rocm`), or empty for CPU since there is no `cpu` feature:

```bash
[ "$DEVICE" = "cpu" ] && FEAT= || FEAT="$DEVICE"
```

Then run exactly:

```bash
make test.llms TEST_DEVICE="$FEAT"
```

## Step 3: rules during the run

- **Per-test 30 s budget.** Flag any individual test that exceeds 30 s wall-clock. That is a real bug — production or test — not flakiness.

## Step 4: report

After all suites finish, sum up the results in an actionable report.

