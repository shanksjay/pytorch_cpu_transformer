# Debug.md — LLDB walkthrough: verify SDPA Query tensor values on Apple Silicon (CPU)

This guide walks through **end-to-end, reproducible debugging** of the **Query (Q) tensor** passed into PyTorch’s **CPU SDPA FlashAttention kernel**, using **LLDB** on Apple Silicon.

It uses **real register mappings, addresses, and memory offsets** captured from this repo’s runs (see `README.md`) and shows how to:
1. Break on the right kernel entrypoint in `libtorch_cpu.dylib`
2. Identify which registers hold `Query`, `Key`, and `Value`
3. Follow the pointer chain `at::Tensor` → `TensorImpl` → data pointer
4. Verify **shape**, **strides**, and **raw float values** for **Query**

Target kernel (as shown in the repo’s LLDB traces):  
`at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(...)` fileciteturn1file2L18-L23

---

## 0) Preconditions

### Ensure the model is actually calling SDPA (FlashAttention CPU)

In `transformer_llama.py`, the attention implementation prefers SDPA when available and uses the **fast causal path** (`attn_mask is None`, `start_pos==0`, `Tq==Tk`) to avoid materializing a full mask. fileciteturn1file5L29-L44

Run:

```bash
python transformer_llama.py --device cpu --profile
```

You should see `aten::_scaled_dot_product_flash_attention_for_cpu` in the profiler output.

---

## 1) Launch under LLDB and break inside the FlashAttention CPU entrypoint

### A) Start LLDB on the interpreter

```bash
lldb python -- transformer_llama.py --device cpu --profile
```

The repo notes show the same approach using `target create` + `settings set -- target.run-args`. fileciteturn1file2L7-L11

### B) Set a regex breakpoint

```lldb
(lldb) breakpoint set -r "scaled_dot_product_flash_attention"
```

It may be “pending” until `libtorch_cpu.dylib` loads, then resolves to many locations. fileciteturn1file2L11-L18

### C) Run

```lldb
(lldb) run
```

You should stop at a frame like:

`libtorch_cpu.dylib` `at::_ops::_scaled_dot_product_flash_attention_for_cpu::call(...)` fileciteturn1file2L18-L23

---

## 2) Confirm you’re stopped at the correct call site

Print a backtrace:

```lldb
(lldb) bt
```

Expected stack shape (from repo notes): fileciteturn1file2L26-L33
- frame 0: `_scaled_dot_product_flash_attention_for_cpu::call`
- frame 1: `at::native::scaled_dot_product_attention`
- frame 2: `_ops::scaled_dot_product_attention::call`
- frame 3+: Python binding glue

---

## 3) Identify Query/Key/Value via registers (ARM64 ABI)

On Apple Silicon (AArch64), the first arguments are passed in:
- `x0`, `x1`, `x2`, ...

In the repo trace, the kernel’s first three `Tensor const&` args correspond to:
- `Query` in `x0`
- `Key` in `x1`
- `Value` in `x2` fileciteturn1file0L8-L12

Dump them:

```lldb
(lldb) register read x0 x1 x2
```

These register values point to stack addresses where the `at::Tensor` references live (not the raw data).

---

## 4) Follow `at::Tensor` → `TensorImpl`

From the repo notes, one working method is to read memory at the tensor reference address and locate the `Impl` pointer. fileciteturn1file0L8-L12

Do it for Query (`x0`):

```lldb
(lldb) memory read --size 8 --format x --count 6 $x0
```

In the repo run, Query’s Impl pointer was `0x0000000aa05c7000` (example). fileciteturn1file0L10-L12

Call that address:

- `Q_IMPL = <value you find>`

> Your run will have different addresses, but the *process* and offsets below are what you’re validating.

---

## 5) Decode `TensorImpl` to get shape, strides, and the data pointer

Dump a block of `Q_IMPL` as 8-byte words (decimal is convenient for sizes/strides):

```lldb
(lldb) memory read --size 8 --format decimal --count 20 Q_IMPL
```

The repo did exactly this to interpret Query tensor metadata. fileciteturn1file0L13-L35

### A) Shape (sizes)

The repo decoding shows:
- Rank is found at offset `+0x40`
- Sizes begin at offset `+0x48`

And the values decode to:

Query shape = `[1, 8, 2048, 64]` fileciteturn1file1L3-L9

### B) Strides

Immediately after sizes, the repo found strides:

`1048576, 64, 512, 1` fileciteturn1file1L11-L15

### C) Data pointer

The repo notes show the (implementation) data pointer address at offset `+0x38` and gives an example:

`Data Pointer Address ... -> 0x10a865630` fileciteturn1file1L16-L17

Read yours:

```lldb
(lldb) memory read --size 8 --format x --count 1 (Q_IMPL + 0x38)
```

Call the returned value:

- `Q_DATA = <pointer you read>`

---

## 6) Dump raw Query tensor values (float32)

Assuming float32 (default for this repo’s CPU path), dump 16 floats:

```lldb
(lldb) memory read --format float --count 16 Q_DATA
```

Repo example output (demonstrates the exact command and sample values): fileciteturn1file1L20-L29

At this point you’ve verified:
- you’re reading Query’s actual buffer
- you can inspect specific positions and validate for NaNs/Inf/corruption

---

## 7) Cross-check against the Python implementation (sanity)

In `transformer_llama.py`, Query is produced by:
- linear projection `q = self.q_proj(x)`
- reshape to `[B, H, T, Dh]` using `einops.rearrange`
- then RoPE is applied to q/k fileciteturn1file6L1-L13

That matches the TensorImpl-decoded shape `[1, 8, 2048, 64]`. fileciteturn1file1L3-L9

---

## 8) Repeat for Key and Value

The repo includes a summary stating that Key/Value can be decoded using the same offsets. fileciteturn1file1L38-L44

Repeat steps 4–6 for:
- `x1` → Key
- `x2` → Value

---

## 9) Optional deep dive: instruction stream and register flow

When stopped at the entrypoint, inspect the current frame’s instructions:

```lldb
(lldb) disassemble --frame
```

Or disassemble from PC:

```lldb
(lldb) register read pc sp
(lldb) disassemble --start-address $pc --count 40
```

Single-step and observe register movement:

```lldb
(lldb) si
(lldb) register read x0 x1 x2 x19 x20 x21 x22 x23
```

This is useful if the compiler copies args into callee-saved registers before deeper calls.

---

## 10) Troubleshooting

### Breakpoint “pending” / “no locations”
Normal until `libtorch_cpu.dylib` loads; run once and LLDB will resolve locations. fileciteturn1file2L11-L18

### Not hitting FlashAttention CPU
Ensure SDPA is enabled and you are on the fast-path conditions (`attn_mask is None`, `start_pos==0`). fileciteturn1file5L36-L44

### Values look wrong
If you’re using float16/bfloat16, dumping as `--format float` will be misleading. Dump bytes/hex and decode accordingly.

---

## Quick command checklist (copy/paste)

Once stopped at `_scaled_dot_product_flash_attention_for_cpu::call`:

```lldb
bt
register read x0 x1 x2
memory read --size 8 --format x --count 6 $x0          # find Q_IMPL
memory read --size 8 --format x --count 1 (Q_IMPL+0x38) # find Q_DATA
memory read --format float --count 16 Q_DATA            # dump values
```
