---
name: run-coverage
description: Runs code coverage checker on the fastest available device. Use when the user asks to run the coverage, or to check the code coverage.
---

# Checking the code coverage

Run every instrumented test suite in the workspace, picking the fastest compiled device backend for the host, then make sure everything is within required limits.

Makefile is the source of truth for the gated values, and the code coverage setup.

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
make coverage TEST_DEVICE="$FEAT"
```

## Step 4: report

After all suites finish, sum up the results in an actionable report. Make sure all code coverage gates are met.


