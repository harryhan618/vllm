# vLLM v1 Engine: Token Sampling Logic Flow

**Version:** vLLM v0.10.2
**Date:** 2025-11-17
**Overview:** Complete request flow from `async_llm.py` to token sampling in the v1 engine

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Detailed Flow Path](#detailed-flow-path)
3. [Per-Request Sampling Metadata](#per-request-sampling-metadata)
4. [Complete Flow Diagram](#complete-flow-diagram)
5. [Key Design Decisions](#key-design-decisions)
6. [Code References](#code-references)

---

## High-Level Architecture

The v1 engine uses a **multi-process architecture** with the following key components:

| Component | File | Description |
|-----------|------|-------------|
| **AsyncLLM** | `vllm/v1/engine/async_llm.py:52` | Main async interface running in the API server process |
| **EngineCoreClient** | `vllm/v1/engine/core_client.py:49` | IPC client communicating via ZMQ |
| **EngineCore** | `vllm/v1/engine/core.py:62` | Core execution loop in a separate process |
| **GPUModelRunner** | `vllm/v1/worker/gpu_model_runner.py:155` | Model execution and sampling |
| **Sampler** | `vllm/v1/sample/sampler.py:22` | Token sampling logic |
| **TopKTopPSampler** | `vllm/v1/sample/ops/topk_topp_sampler.py:24` | Top-k/top-p filtering and sampling |

---

## Detailed Flow Path

### 1. Request Entry: AsyncLLM.generate()

**Location:** `vllm/v1/engine/async_llm.py:332`

```python
async def generate(self, prompt, sampling_params, request_id, ...):
    # Process input → EngineCoreRequest
    q = await self.add_request(request_id, prompt, sampling_params, ...)

    # Yield outputs from queue
    while not finished:
        out = await q.get()
        yield out
```

**Key actions:**
- Converts user input to `EngineCoreRequest` via `Processor.process_inputs()` (line 293)
- Adds request to `OutputProcessor` (line 318)
- Sends request to EngineCore via ZMQ (line 322)
- Returns a `RequestOutputCollector` queue for streaming results

---

### 2. Background Output Handler

**Location:** `vllm/v1/engine/async_llm.py:444`

A background asyncio task continuously pulls outputs from EngineCore:

```python
async def output_handler():
    while True:
        # 1. Pull EngineCoreOutputs from EngineCore (separate process)
        outputs = await engine_core.get_output_async()
        num_outputs = len(outputs.outputs)

        # 2. Process EngineCoreOutputs
        processed_outputs = output_processor.process_outputs(
            outputs.outputs, outputs.timestamp, iteration_stats
        )
        # NOTE: RequestOutputs are pushed to their queues

        # 3. Abort any requests that finished due to stop strings
        await engine_core.abort_requests_async(
            processed_outputs.reqs_to_abort
        )

        # 4. Log stats
        if logger_manager:
            logger_manager.record(
                engine_idx=outputs.engine_index,
                scheduler_stats=outputs.scheduler_stats,
                iteration_stats=iteration_stats,
            )
```

**Flow:**
- Runs in background (`asyncio.create_task`)
- Pulls outputs from EngineCore (via ZMQ)
- `OutputProcessor` detokenizes and pushes to per-request queues
- These queues feed the `generate()` async iterator

---

### 3. IPC Layer: EngineCoreClient

**AsyncMPClient** (`vllm/v1/engine/core_client.py:759`) manages communication:

- **Input socket** (zmq.ROUTER): Sends `EngineCoreRequest` to EngineCore
- **Output socket** (zmq.PULL): Receives `EngineCoreOutputs` from EngineCore
- Uses msgpack serialization for efficient transfer

**add_request_async()** (line 902):
```python
async def add_request_async(self, request: EngineCoreRequest) -> None:
    request.client_index = self.client_index
    await self._send_input(EngineCoreRequestType.ADD, request)
    self._ensure_output_queue_task()
```

**get_output_async()** (line 837):
```python
async def get_output_async(self) -> EngineCoreOutputs:
    self._ensure_output_queue_task()
    outputs = await self.outputs_queue.get()  # Filled by background task
    if isinstance(outputs, Exception):
        raise self._format_exception(outputs) from None
    return outputs
```

---

### 4. EngineCore Execution Loop

**Location:** `vllm/v1/engine/core.py:280`

The `step()` method is the heart of the execution:

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.

    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """
    # Check for any requests remaining in the scheduler
    if not self.scheduler.has_requests():
        return {}, False

    # 1. Schedule requests (select which tokens to compute)
    scheduler_output = self.scheduler.schedule()

    # 2. Execute model (forward pass + sampling)
    model_output = self.execute_model_with_error_logging(
        self.model_executor.execute_model,
        scheduler_output
    )

    # 3. Update scheduler & create outputs
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return (engine_core_outputs,
            scheduler_output.total_num_scheduled_tokens > 0)
```

This runs in a busy loop in a separate process (`EngineCoreProc`).

---

### 5. Model Execution: GPUModelRunner.execute_model()

**Location:** `vllm/v1/worker/gpu_model_runner.py:2000`

```python
def execute_model(self, scheduler_output, ...):
    # ===== 1. PREPROCESS =====
    with record_function_or_nullcontext("Preprocess"):
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        # Prepare the decoder inputs
        (attn_metadata, logits_indices, spec_decode_metadata,
         num_scheduled_tokens_np, spec_decode_common_attn_metadata,
         max_query_len) = self._prepare_inputs(scheduler_output)

        (num_scheduled_tokens, num_input_tokens, num_tokens_across_dp,
         input_ids, inputs_embeds, positions, intermediate_tensors,
         model_kwargs) = self._preprocess(scheduler_output, intermediate_tensors)

        uniform_decode = (max_query_len == self.uniform_decode_query_len) and \
                         (num_scheduled_tokens == self.input_batch.num_reqs * max_query_len)
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                           uniform_decode=uniform_decode)
        cudagraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(batch_descriptor)

    # ===== 2. FORWARD =====
    with (set_forward_context(...), record_function_or_nullcontext("Forward")):
        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    # ===== 3. POSTPROCESS =====
    with record_function_or_nullcontext("Postprocess"):
        hidden_states = model_output

        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states
            return hidden_states

        if self.is_pooling_model:
            return self._pool(hidden_states, ...)

        # Extract logits only for tokens that need sampling
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

    # ===== 4. SAMPLE =====
    with record_function_or_nullcontext("Sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)

    # ===== 5. BOOKKEEPING =====
    with record_function_or_nullcontext("Bookkeep"):
        (num_nans_in_logits, logprobs_lists, valid_sampled_token_ids,
         prompt_logprobs_dict, req_ids_output_copy, req_id_to_index_output_copy,
         invalid_req_indices) = self._bookkeeping_sync(
            scheduler_output, sampler_output, logits, hidden_states,
            num_scheduled_tokens
        )

    # ===== 6. DRAFT (Speculative Decoding) =====
    if self.speculative_config:
        with record_function_or_nullcontext("Draft"):
            self._draft_token_ids = self.propose_draft_token_ids(...)

    # ===== 7. EPLB (Expert Parallelism Load Balancing) =====
    with record_function_or_nullcontext("EPLB"):
        self.eplb_step()

    # ===== 8. RETURN OUTPUT =====
    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=[],
        kv_connector_output=kv_connector_output,
        num_nans_in_logits=num_nans_in_logits,
    )

    if not self.use_async_scheduling:
        return output

    return AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=sampler_output.sampled_token_ids,
        invalid_req_indices=invalid_req_indices,
        async_output_copy_stream=self.async_output_copy_stream,
    )
```

**Key insight:** Only tokens at `logits_indices` are computed (selected by scheduler based on which tokens need sampling).

---

### 6. Token Sampling: GPUModelRunner._sample()

**Location:** `vllm/v1/worker/gpu_model_runner.py:1828`

```python
def _sample(self, logits, spec_decode_metadata):
    # Sample the next token and get logprobs if needed
    sampling_metadata = self.input_batch.sampling_metadata

    if spec_decode_metadata is None:
        # Standard sampling
        sampler_output = self.sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
    else:
        # Speculative decoding sampling
        bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
        sampler_output = self.sampler(
            logits=bonus_logits,
            sampling_metadata=sampling_metadata,
        )
        bonus_token_ids = sampler_output.sampled_token_ids

        target_logits = logits[spec_decode_metadata.target_logits_indices]
        output_token_ids = self.rejection_sampler(
            spec_decode_metadata,
            None,  # draft_probs
            target_logits,
            bonus_token_ids,
            sampling_metadata,
        )
        sampler_output.sampled_token_ids = output_token_ids
        self._update_states_after_model_execute(output_token_ids)

    return sampler_output
```

Delegates to the `Sampler` module.

---

### 7. Sampler Forward Pass

**Location:** `vllm/v1/sample/sampler.py:70`

The `Sampler.forward()` method performs sampling in **9 steps**:

```python
def forward(self, logits, sampling_metadata):
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested, compute raw logprobs or clone logits
    2. Convert logits to float32
    3. Apply allowed token ids whitelist
    4. Apply bad words exclusion
    5. Apply logit processors (non-argmax-invariant)
    6. Apply penalties
    7. Sample the next tokens
    8. Gather logprobs of top-k and sampled token
    9. Return SamplerOutput
    """

    # ===== STEP 1: Compute raw logprobs (if requested) =====
    num_logprobs = sampling_metadata.max_num_logprobs
    if num_logprobs is not None:
        if self.logprobs_mode == LogprobsMode.RAW_LOGPROBS:
            raw_logprobs = self.compute_logprobs(logits)  # log_softmax
        elif self.logprobs_mode == LogprobsMode.RAW_LOGITS:
            raw_logprobs = logits.clone()

    # ===== STEP 2: Convert to float32 =====
    logits = logits.to(torch.float32)

    # ===== STEP 3: Apply allowed token IDs whitelist =====
    logits = self.apply_allowed_token_ids(logits, sampling_metadata)

    # ===== STEP 4: Apply bad words exclusion =====
    logits = self.apply_bad_words(logits, sampling_metadata)

    # ===== STEP 5: Apply non-argmax-invariant logit processors =====
    # (min_tokens, logit_bias, etc.)
    for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
        logits = processor.apply(logits)

    # ===== STEP 6: Apply penalties =====
    # (repetition, frequency, presence penalties)
    logits = self.apply_penalties(logits, sampling_metadata)

    # ===== STEP 7: Sample the next token =====
    sampled, processed_logprobs = self.sample(logits, sampling_metadata)
    if processed_logprobs is not None:
        raw_logprobs = processed_logprobs

    # Convert to int64 for compatibility
    sampled = sampled.long()

    # ===== STEP 8: Gather top-k logprobs and sampled token logprobs =====
    logprobs_tensors = None if num_logprobs is None else \
        self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)

    # Use int32 to reduce tensor size
    sampled = sampled.to(torch.int32)

    # ===== STEP 9: Return SamplerOutput =====
    sampler_output = SamplerOutput(
        # Expand to 2D: [num_requests, 1]
        sampled_token_ids=sampled.unsqueeze(-1),
        logprobs_tensors=logprobs_tensors,
    )
    return sampler_output
```

---

### 8. Core Sampling Logic: Sampler.sample()

**Location:** `vllm/v1/sample/sampler.py:139`

The `sample()` method handles greedy vs random sampling:

```python
def sample(self, logits, sampling_metadata):
    """Sample logits based on sampling metadata.

    The various logits processing functions called in this method
    may update the logits tensor in-place.
    """

    assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)

    # ===== GREEDY PATH =====
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = self.greedy_sample(logits)  # argmax
        if sampling_metadata.all_greedy:
            processed_logprobs = None
            if sampling_metadata.max_num_logprobs is not None:
                if self.logprobs_mode == LogprobsMode.PROCESSED_LOGITS:
                    processed_logprobs = logits
                elif self.logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS:
                    processed_logprobs = self.compute_logprobs(logits)
            return greedy_sampled, processed_logprobs

    # ===== RANDOM PATH =====
    assert sampling_metadata.temperature is not None

    # a) Apply temperature
    logits = self.apply_temperature(logits, sampling_metadata.temperature)

    # b) Apply argmax-invariant logit processors (min_p, etc.)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    # c) Apply top-k and/or top-p, then sample
    random_sampled, processed_logprobs = self.topk_topp_sampler(
        logits,
        sampling_metadata.generators,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )

    # d) Mix greedy and random based on temperature threshold
    if greedy_sampled is None:
        return random_sampled, processed_logprobs

    sampled = torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,  # 1e-5
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,  # Reuse tensor
    )
    return sampled, processed_logprobs
```

**Temperature application** (line 128):
```python
def apply_temperature(self, logits, temp):
    # Use in-place division to avoid creating a new tensor
    return logits.div_(temp.unsqueeze(dim=1))
```

**Greedy sampling** (line 136):
```python
def greedy_sample(self, logits):
    return logits.argmax(dim=-1).view(-1)
```

---

### 9. Top-K/Top-P Sampling: TopKTopPSampler

**Location:** `vllm/v1/sample/ops/topk_topp_sampler.py:24`

The sampler has **two implementations**:

#### A. PyTorch Native Implementation

**Location:** `topk_topp_sampler.py:79`

```python
def forward_native(self, logits, generators, k, p):
    """
    PyTorch-native implementation of top-k and top-p sampling.

    The logits tensor may be updated in-place.
    """
    # Apply top-k/top-p masking
    logits = self.apply_top_k_top_p(logits, k, p)

    # Prepare logprobs to return (if requested)
    logits_to_return = None
    if self.logprobs_mode == LogprobsMode.PROCESSED_LOGITS:
        logits_to_return = logits
    elif self.logprobs_mode == LogprobsMode.PROCESSED_LOGPROBS:
        logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

    # Convert to probabilities
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # Random sampling using Gumbel-max trick
    return random_sample(probs, generators), logits_to_return
```

**random_sample()** (line 194):
```python
def random_sample(probs, generators):
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = torch.empty_like(probs)

    # Batch process requests without their own seeds
    if len(generators) != probs.shape[0]:
        q.exponential_()  # Sample from Exp(1) = Gumbel distribution

    # Overwrite values for requests with their own seeds
    if generators:
        # TODO(woosuk): This can be slow because we handle each request
        # one by one. Optimize this.
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

    # Gumbel-max trick: argmax(log(p) - log(-log(u))) = argmax(p / u)
    # where u ~ Exp(1)
    return probs.div_(q).argmax(dim=-1).view(-1)
```

**apply_top_k_top_p()** (line 126):
```python
def apply_top_k_top_p(logits, k, p):
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits
        # Avoid sorting vocab for top-k only case
        return apply_top_k_only(logits, k)

    # Sort logits in descending order (actually ascending, then reverse indexed)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # Keep at least one token
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities back to original order
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits
```

**apply_top_k_only()** (line 169):
```python
def apply_top_k_only(logits, k):
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()

    # topk.values tensor has shape [batch_size, max_top_k]
    # Convert top k to 0-based index in range [0, max_top_k)
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())

    # Handle non-topk rows
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits
```

#### B. FlashInfer Optimized Implementation

**Location:** `topk_topp_sampler.py:100`

```python
def forward_cuda(self, logits, generators, k, p):
    """More optimized implementation for top-k and top-p sampling."""
    # We prefer `random_sample` over `flashinfer_sample` when sorting is
    # not needed. This is because `random_sample` does not require
    # CPU-GPU synchronization while `flashinfer_sample` does.
    if (k is None and p is None) or generators:
        if generators:
            logger.warning_once("FlashInfer 0.2.3+ does not support "
                                "per-request generators. Falling back to "
                                "PyTorch-native implementation.")
        return self.forward_native(logits, generators, k, p)

    assert self.logprobs_mode not in (
        LogprobsMode.PROCESSED_LOGITS, LogprobsMode.PROCESSED_LOGPROBS
    ), "FlashInfer does not support returning logits/logprobs"

    # flashinfer sampling functions expect contiguous logits
    # In flex_attn/triton_attn fp32 inference, logits can be non-contiguous
    # because of slicing operation in logits_processor
    return flashinfer_sample(logits.contiguous(), k, p, generators), None
```

**flashinfer_sample()** (line 218):
```python
def flashinfer_sample(logits, k, p, generators):
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.

    NOTE: This function includes CPU-GPU synchronization, while `random_sample`
    does not. Call this function at the end of the forward pass to minimize
    the synchronization overhead.
    """
    assert not (k is None and p is None)

    if k is None:
        # Top-p only
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
            probs, p, deterministic=True
        )
    elif p is None:
        # Top-k only
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
            probs, k, deterministic=True
        )
    else:
        # Both top-k and top-p
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            logits, k, p, deterministic=True
        )

    return next_token_ids.view(-1)
```

**Note:** FlashInfer uses C++/CUDA kernels with rejection sampling to avoid sorting, making it much faster for large vocabularies.

---

## Per-Request Sampling Metadata

Each request's sampling parameters are batched in `SamplingMetadata` (`vllm/v1/sample/metadata.py`):

### Key Fields

| Field | Type | Description |
|-------|------|-------------|
| `temperature` | `torch.Tensor` | Per-request temperature values (shape: [batch_size]) |
| `top_k` | `torch.Tensor` | Per-request top-k values (shape: [batch_size]) |
| `top_p` | `torch.Tensor` | Per-request top-p values (shape: [batch_size]) |
| `generators` | `dict[int, torch.Generator]` | Per-request random number generators (for seed control) |
| `logitsprocs` | `LogitsProcessors` | Batched logits processors |
| `all_greedy` | `bool` | True if all requests use greedy sampling (temperature=0) |
| `all_random` | `bool` | True if all requests use random sampling |
| `max_num_logprobs` | `int` | Maximum number of logprobs to return |

### Batched Operations

- **Penalties:** Applied per-request using indexing/masking on token IDs
- **Temperature:** Vectorized division: `logits.div_(temperature.unsqueeze(1))`
- **Top-k/top-p:** Handles heterogeneous values per request via tensor operations
- **Sampling:** Uses per-request generators where needed (via dict lookup)

### Example

For a batch of 3 requests:
```python
sampling_metadata = SamplingMetadata(
    temperature=torch.tensor([0.0, 0.7, 1.0]),  # req0: greedy, req1/2: random
    top_k=torch.tensor([50, 40, 0]),             # req0/1: top-k, req2: no top-k
    top_p=torch.tensor([0.95, 0.9, 0.0]),        # req0/1: top-p, req2: no top-p
    generators={1: torch.Generator().manual_seed(42)},  # req1 has custom seed
    all_greedy=False,
    all_random=False,
    ...
)
```

**Execution:**
1. Greedy sampling applied to all → `greedy_sampled = logits.argmax(dim=-1)`
2. Temperature applied: `logits /= temperature.unsqueeze(1)`
3. Top-k/top-p masking applied per-request
4. Random sampling with Gumbel-max trick
5. Final output: `torch.where(temperature < 1e-5, greedy_sampled, random_sampled)`

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Request                                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ AsyncLLM.generate() [async_llm.py:332]                              │
│  ├─ Processor.process_inputs() → EngineCoreRequest                  │
│  ├─ OutputProcessor.add_request()                                   │
│  └─ EngineCoreClient.add_request_async() → ZMQ send                 │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ EngineCoreProc (separate process) [core.py]                         │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ EngineCore.step() [core.py:280] - Busy loop                    │ │
│  │                                                                  │ │
│  │  1. Scheduler.schedule()                                        │ │
│  │     └─ Select requests, batch them, determine tokens to sample │ │
│  │                                                                  │ │
│  │  2. Executor.execute_model(scheduler_output)                    │ │
│  │     └─ Calls GPUModelRunner.execute_model()                     │ │
│  │                                                                  │ │
│  │  3. Scheduler.update_from_output()                              │ │
│  │     └─ Create EngineCoreOutputs                                 │ │
│  │                                                                  │ │
│  │  4. ZMQ send → EngineCoreOutputs                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ GPUModelRunner.execute_model() [gpu_model_runner.py:2000]          │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 1. PREPROCESS                                                   │ │
│  │    ├─ _prepare_inputs() → attention metadata, logits_indices   │ │
│  │    └─ _preprocess() → input_ids, positions, embeddings         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 2. FORWARD                                                      │ │
│  │    └─ model(input_ids, positions, ...) → hidden_states         │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 3. POSTPROCESS                                                  │ │
│  │    ├─ hidden_states[logits_indices] → sample_hidden_states     │ │
│  │    ├─ model.compute_logits() → logits                          │ │
│  │    └─ apply_grammar_bitmask() (if structured output)           │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 4. SAMPLE                                                       │ │
│  │    └─ _sample(logits, spec_decode_metadata)                    │ │
│  │       └─ sampler(logits, sampling_metadata)                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 5. BOOKKEEPING                                                  │ │
│  │    ├─ Extract sampled_token_ids (GPU → CPU sync)               │ │
│  │    ├─ Compute prompt logprobs                                  │ │
│  │    └─ Handle partial prefills                                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ 6. RETURN ModelRunnerOutput                                     │ │
│  │    └─ (sampled_token_ids, logprobs, req_ids, ...)             │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Sampler.forward() [sampler.py:70]                                  │
│                                                                       │
│  1. Compute raw logprobs (if requested)                             │
│     └─ logits.log_softmax(dim=-1)                                  │
│                                                                       │
│  2. Convert to float32                                              │
│                                                                       │
│  3. Apply allowed token IDs whitelist                               │
│                                                                       │
│  4. Apply bad words exclusion                                       │
│                                                                       │
│  5. Apply non-argmax-invariant processors                           │
│     └─ min_tokens, logit_bias                                      │
│                                                                       │
│  6. Apply penalties                                                  │
│     └─ repetition, frequency, presence                             │
│                                                                       │
│  7. Sample the next token                                           │
│     └─ sample(logits, sampling_metadata)                           │
│                                                                       │
│  8. Gather logprobs of top-k and sampled token                      │
│                                                                       │
│  9. Return SamplerOutput                                            │
│     └─ (sampled_token_ids, logprobs_tensors)                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Sampler.sample() [sampler.py:139]                                  │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ GREEDY PATH (temperature < 1e-5 or all_greedy=True)            │ │
│  │  └─ greedy_sampled = logits.argmax(dim=-1)                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ RANDOM PATH (temperature ≥ 1e-5)                                │ │
│  │                                                                  │ │
│  │  a) Apply temperature                                           │ │
│  │     └─ logits /= temperature.unsqueeze(1)                      │ │
│  │                                                                  │ │
│  │  b) Apply argmax-invariant processors                           │ │
│  │     └─ min_p, etc.                                             │ │
│  │                                                                  │ │
│  │  c) TopKTopPSampler(logits, generators, top_k, top_p)          │ │
│  │     ├─ Apply top-k mask                                         │ │
│  │     ├─ Apply top-p mask                                         │ │
│  │     ├─ softmax → probs                                          │ │
│  │     └─ random_sample (Gumbel-max trick)                        │ │
│  │                                                                  │ │
│  │  d) random_sampled                                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ MIX GREEDY & RANDOM                                             │ │
│  │  └─ torch.where(temperature < 1e-5, greedy, random)            │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ TopKTopPSampler [topk_topp_sampler.py]                             │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ IMPLEMENTATION SELECTION                                        │ │
│  │  ├─ forward_native() - PyTorch implementation                   │ │
│  │  └─ forward_cuda() - FlashInfer optimized (if available)       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ forward_native() [line 79]                                      │ │
│  │                                                                  │ │
│  │  1. apply_top_k_top_p(logits, k, p)                            │ │
│  │     ├─ Top-k only: apply_top_k_only()                          │ │
│  │     │  └─ logits.topk(k) → mask values below kth              │ │
│  │     └─ Top-p: sort + cumsum + mask                             │ │
│  │                                                                  │ │
│  │  2. probs = logits.softmax(dim=-1)                             │ │
│  │                                                                  │ │
│  │  3. random_sample(probs, generators)                            │ │
│  │     ├─ q = torch.empty_like(probs).exponential_()              │ │
│  │     └─ return (probs / q).argmax(dim=-1)                       │ │
│  │        (Gumbel-max trick)                                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ forward_cuda() [line 100] - FlashInfer                         │ │
│  │                                                                  │ │
│  │  flashinfer_sample(logits.contiguous(), k, p, generators)       │ │
│  │  ├─ Top-p only: flashinfer.sampling.top_p_sampling_from_probs │ │
│  │  ├─ Top-k only: flashinfer.sampling.top_k_sampling_from_probs │ │
│  │  └─ Both: flashinfer.sampling.top_k_top_p_sampling_from_logits│ │
│  │                                                                  │ │
│  │  Note: Uses rejection sampling in C++/CUDA kernels              │ │
│  │        Faster, but includes CPU-GPU sync                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Sampled Token IDs [batch_size, 1]                                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ModelRunnerOutput → Scheduler → EngineCoreOutputs                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼ (ZMQ receive)
┌─────────────────────────────────────────────────────────────────────┐
│ OutputProcessor.process_outputs() [output_processor.py]            │
│  ├─ Detokenize sampled_token_ids                                    │
│  ├─ Process logprobs                                                 │
│  └─ Push to RequestOutputCollector queue                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ AsyncLLM.generate() yields RequestOutput                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     User Receives Output                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Multi-Process Isolation
**Benefit:** EngineCore runs in a separate process for stability (crash isolation). If model execution crashes, the API server remains alive.

### 2. Async I/O with ZMQ
**Benefit:** Non-blocking communication between AsyncLLM and EngineCore using ZMQ sockets + asyncio.

### 3. Batched Sampling
**Benefit:** All per-request parameters are batched into tensors for GPU efficiency. Operations like temperature scaling, top-k/top-p masking, and sampling are vectorized.

### 4. Selective Logit Computation
**Benefit:** Only `logits_indices` tokens are sampled (not all tokens in the batch). Saves compute for partial prefills.

### 5. FlashInfer Optimization
**Benefit:** Uses efficient CUDA kernels for top-k/top-p sampling when available. Faster than PyTorch's native implementation via rejection sampling.

### 6. Gumbel-Max Trick
**Benefit:** Avoids slow `torch.multinomial` (which causes CPU-GPU sync). Instead:
```python
q = torch.empty_like(probs).exponential_()  # Sample from Exp(1)
sampled = (probs / q).argmax(dim=-1)        # Equivalent to multinomial
```

### 7. In-Place Operations
**Benefit:** Logits are modified in-place during temperature scaling, top-k/top-p masking, and penalty application to save memory.

### 8. Continuous Batching
**Benefit:** Scheduler dynamically batches requests with different lengths/states. Maximizes GPU utilization.

### 9. Separate Background Output Handler
**Benefit:** Async task continuously pulls outputs from EngineCore, allowing the main generate() loop to focus on streaming results to users.

### 10. Per-Request Generators
**Benefit:** Supports custom random seeds per request via `generators` dict. Enables reproducible sampling when needed.

---

## Code References

### Main Files

| File | Lines | Description |
|------|-------|-------------|
| `vllm/v1/engine/async_llm.py` | 52-762 | AsyncLLM main interface, generate() method, output_handler |
| `vllm/v1/engine/core_client.py` | 49-1334 | EngineCoreClient, AsyncMPClient, ZMQ communication |
| `vllm/v1/engine/core.py` | 62-600 | EngineCore, step() execution loop |
| `vllm/v1/worker/gpu_model_runner.py` | 155-2300 | GPUModelRunner, execute_model(), _sample() |
| `vllm/v1/sample/sampler.py` | 22-270 | Sampler, forward(), sample() |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | 24-255 | TopKTopPSampler, random_sample(), flashinfer_sample() |
| `vllm/v1/sample/metadata.py` | - | SamplingMetadata definition |
| `vllm/v1/sample/ops/penalties.py` | - | Penalty application functions |
| `vllm/v1/sample/ops/bad_words.py` | - | Bad words filtering |

### Key Function References

| Function | Location | Description |
|----------|----------|-------------|
| `AsyncLLM.generate()` | `async_llm.py:332` | Main entry point for text generation |
| `AsyncLLM.add_request()` | `async_llm.py:270` | Adds request to engine and output processor |
| `AsyncLLM.output_handler()` | `async_llm.py:444` | Background task pulling outputs from EngineCore |
| `AsyncMPClient.add_request_async()` | `core_client.py:902` | Sends request via ZMQ |
| `AsyncMPClient.get_output_async()` | `core_client.py:837` | Receives outputs via ZMQ |
| `EngineCore.step()` | `core.py:280` | Main execution loop step |
| `GPUModelRunner.execute_model()` | `gpu_model_runner.py:2000` | Model forward + sampling |
| `GPUModelRunner._sample()` | `gpu_model_runner.py:1828` | Delegates to Sampler |
| `Sampler.forward()` | `sampler.py:70` | 9-step sampling pipeline |
| `Sampler.sample()` | `sampler.py:139` | Greedy vs random sampling logic |
| `TopKTopPSampler.forward_native()` | `topk_topp_sampler.py:79` | PyTorch top-k/top-p + sampling |
| `TopKTopPSampler.forward_cuda()` | `topk_topp_sampler.py:100` | FlashInfer optimized sampling |
| `random_sample()` | `topk_topp_sampler.py:194` | Gumbel-max sampling trick |
| `flashinfer_sample()` | `topk_topp_sampler.py:218` | FlashInfer CUDA kernels |
| `apply_top_k_top_p()` | `topk_topp_sampler.py:126` | Top-k/top-p masking |

---

## Appendix: Gumbel-Max Trick Explanation

The Gumbel-max trick is a method to sample from a categorical distribution without using `torch.multinomial`:

**Mathematical equivalence:**
```
Sample i ~ Categorical(p₁, ..., pₙ)
≡
i = argmax_j(log(pⱼ) + Gumbel(0, 1))
≡
i = argmax_j(log(pⱼ) - log(-log(U)))  where U ~ Uniform(0, 1)
≡
i = argmax_j(pⱼ / E)  where E ~ Exp(1)
```

**Implementation:**
```python
q = torch.empty_like(probs).exponential_()  # Sample E ~ Exp(1)
sampled = (probs / q).argmax(dim=-1)        # argmax(p / E)
```

**Benefits:**
- No CPU-GPU synchronization (unlike `torch.multinomial`)
- Fully parallelizable on GPU
- Numerically stable

---

**End of Document**
