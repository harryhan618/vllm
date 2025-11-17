# SamplingParams to LogitsProcessor Data Flow in vLLM v0.10.2

## Overview

This document explains how `SamplingParams` from a request flows through the vLLM v0.10.2 engine to reach individual `LogitsProcessor` instances, specifically `ReplayLogitsProcessor`.

## Complete Data Flow

```
AsyncLLM.add_request(params: SamplingParams)
    ↓
Processor.process_inputs(params)
    ↓
EngineCoreRequest(sampling_params=params.clone())
    ↓
Request.from_engine_core_request()
    ↓
Request.sampling_params
    ↓
Scheduler.add_request(request)
    ↓
NewRequestData.from_request()
    ↓
SchedulerOutput.scheduled_new_reqs[].sampling_params
    ↓
GPUModelRunner._update_states()
    ↓
CachedRequestState(sampling_params=new_req_data.sampling_params)
    ↓
InputBatch.add_request(req_state)
    ↓
InputBatch._register_add_request()
    ↓
BatchUpdateBuilder.added.append((index, sampling_params, prompt_ids, output_ids))
    ↓
InputBatch.refresh_metadata()
    ↓
BatchUpdateBuilder.get_and_reset() → BatchUpdate
    ↓
LogitsProcessor.update_state(batch_update)
    ↓
AdapterLogitsProcessor.update_state()
    ↓
process_dict_updates(req_info, batch_update, _new_state)
    ↓
AdapterLogitsProcessor._new_state(params, prompt_ids, output_ids)
    ↓
ReplayLogitsProcessor.new_req_logits_processor(params: SamplingParams)
    ↓
params.extra_args.get("replay_token_ids")
    ↓
_ReplayRequestProcessor(token_ids, force_stop_token_id)
```

---

## Detailed Step-by-Step Breakdown

### 1. Entry Point: AsyncLLM.add_request()

**File**: `vllm/v1/engine/async_llm.py` (lines 270-310)

```python
async def add_request(
    self,
    request_id: str,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],  # <-- INPUT
    arrival_time: Optional[float] = None,
    lora_request: Optional[LoRARequest] = None,
    tokenization_kwargs: Optional[dict[str, Any]] = None,
    trace_headers: Optional[Mapping[str, str]] = None,
    priority: int = 0,
    data_parallel_rank: Optional[int] = None,
) -> RequestOutputCollector:
    # Create a new output collector
    queue = RequestOutputCollector(output_kind=params.output_kind)

    # Convert Input --> Request
    prompt_str, request = self.processor.process_inputs(
        request_id, prompt, params, arrival_time, lora_request,
        tokenization_kwargs, trace_headers, priority, data_parallel_rank)

    # ... handle n>1 case ...

    await self._add_request(request, prompt_str, None, 0, queue)
    return queue
```

### 2. Processor Creates EngineCoreRequest

**File**: `vllm/v1/engine/processor.py` (lines 314-437)

```python
def process_inputs(
    self,
    request_id: str,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],  # <-- RECEIVES
    ...
) -> tuple[Optional[str], EngineCoreRequest]:

    # Validate params
    self._validate_params(params, lora_request)

    # Process inputs (tokenization, multimodal, etc.)
    processed_inputs = self.input_preprocessor.preprocess(...)

    # Clone and prepare sampling params
    sampling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()  # <-- CLONE
        if sampling_params.max_tokens is None:
            sampling_params.max_tokens = (
                self.model_config.max_model_len -
                len(decoder_inputs["prompt_token_ids"]))
        sampling_params.update_from_generation_config(...)
        sampling_params.update_from_tokenizer(...)

    # Create EngineCoreRequest
    return decoder_inputs.get("prompt"), EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=decoder_inputs["prompt_token_ids"],
        mm_features=mm_features,
        sampling_params=sampling_params,  # <-- STORED
        pooling_params=pooling_params,
        eos_token_id=eos_token_id,
        arrival_time=arrival_time,
        lora_request=lora_request,
        cache_salt=decoder_inputs.get("cache_salt"),
        priority=priority,
        data_parallel_rank=data_parallel_rank,
        trace_headers=trace_headers,
    )
```

### 3. EngineCoreRequest Structure

**File**: `vllm/v1/engine/__init__.py` (lines 43-71)

```python
class EngineCoreRequest(
        msgspec.Struct,
        array_like=True,
        omit_defaults=True,
        gc=False):

    request_id: str
    prompt_token_ids: list[int]
    mm_features: Optional[list[MultiModalFeatureSpec]]
    sampling_params: Optional[SamplingParams]  # <-- PRESERVED
    pooling_params: Optional[PoolingParams]
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]
    cache_salt: Optional[str]
    data_parallel_rank: Optional[int]
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0
    trace_headers: Optional[Mapping[str, str]] = None
```

### 4. Request Wrapper Creation

**File**: `vllm/v1/request.py` (lines 122-144)

```python
@classmethod
def from_engine_core_request(
    cls,
    request: EngineCoreRequest,
    block_hasher: BlockHasher,
) -> "Request":
    # ... hash calculation ...

    return cls(
        request_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_features=request.mm_features,
        sampling_params=request.sampling_params,  # <-- TRANSFERRED
        pooling_params=request.pooling_params,
        eos_token_id=request.eos_token_id,
        arrival_time=request.arrival_time,
        lora_request=request.lora_request,
        block_hash=block_hash,
        cache_salt=request.cache_salt,
        data_parallel_rank=request.data_parallel_rank,
        trace_headers=request.trace_headers,
    )
```

### 5. Scheduler Processing

**File**: `vllm/v1/core/sched/scheduler.py` (lines 1094-1095)

```python
def add_request(self, request: Request) -> None:
    """Add a new request to the scheduler."""
    self.waiting.add_request(request)
    self.requests[request.request_id] = request
```

**File**: `vllm/v1/core/sched/scheduler.py` (lines 568-571)

When scheduling requests, `NewRequestData` is created:

```python
new_reqs_data = [
    NewRequestData.from_request(
        req, req_to_new_blocks[req.request_id].get_block_ids())
    for req in scheduled_new_reqs
]
```

### 6. NewRequestData Creation

**File**: `vllm/v1/core/sched/output.py` (lines 40-56)

```python
@classmethod
def from_request(
    cls,
    request: Request,
    block_ids: tuple[list[int], ...]
) -> "NewRequestData":
    return cls(
        req_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        mm_kwargs=request.mm_kwargs,
        mm_positions=request.mm_positions,
        sampling_params=request.sampling_params,  # <-- PASSED
        pooling_params=request.pooling_params,
        block_ids=block_ids,
        encoder_block_ids=encoder_block_ids,
        lora_request=request.lora_request,
        cache_salt=request.cache_salt,
    )
```

### 7. GPU Model Runner Processing

**File**: `vllm/v1/worker/gpu_model_runner.py` (lines 534-668)

This is where the scheduler output is consumed and requests are added to the batch:

```python
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    ...
) -> Optional[ModelOutput]:
    # Update internal state with scheduler output
    self._update_states(scheduler_output)

    # ... execute forward pass ...

def _update_states(self, scheduler_output: SchedulerOutput):
    """Update batch state based on scheduler output."""

    reqs_to_add: list[CachedRequestState] = []

    # Process new requests
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id

        # Create cached request state
        req_state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            prompt=None,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_placeholders=multi_modal_placeholders,
            sampling_params=new_req_data.sampling_params,  # <-- EXTRACTED
            pooling_params=new_req_data.pooling_params,
            block_ids=new_req_data.block_ids,
            encoder_block_ids=new_req_data.encoder_block_ids,
            lora_request=new_req_data.lora_request,
            cache_salt=new_req_data.cache_salt,
            num_computed_tokens=0,
            output_token_ids=[],  # <-- Empty output list created
        )

        self.requests[req_id] = req_state
        reqs_to_add.append(req_state)

    # Add new requests to the persistent batch
    for request in reqs_to_add:
        self.input_batch.add_request(request)

    # ... handle removed and resumed requests ...

    # Condense batch and refresh metadata
    self.input_batch.condense(scheduler_output.num_scheduled_tokens)
    self.input_batch.refresh_metadata()  # <-- TRIGGERS LOGITS PROCESSOR UPDATE
```

### 8. InputBatch.add_request() - CRITICAL STEP

**File**: `vllm/v1/worker/gpu_input_batch.py` (lines 274-293)

This is where `SamplingParams` enters the logits processor update pipeline:

```python
def add_request(self, request: CachedRequestState) -> int:
    """Add a new request to the batch."""

    # Register the request and get batch index
    req_index = self._register_add_request(request)

    # ... store request data in batch tensors ...

    return req_index

def _register_add_request(self, request: CachedRequestState) -> int:
    """Register a new request for logits processor updates."""

    # Get next available index
    if (new_req_index := self.batch_update_builder.pop_removed()) is None:
        new_req_index = self.num_reqs

    assert new_req_index < self.max_num_reqs
    self.batch_update_builder.batch_changed = True

    if request.sampling_params:
        # Register with BatchUpdateBuilder
        # This is the KEY STEP where SamplingParams is queued for logits processors
        self.batch_update_builder.added.append(
            (new_req_index,                # Batch position
             request.sampling_params,       # <-- SAMPLING PARAMS PASSED
             request.prompt_token_ids,      # Prompt tokens
             request.output_token_ids)      # Reference to output token list
        )

    return new_req_index
```

**Data Structure**: The `added` list contains tuples of type:
```python
AddedRequest = tuple[int, SamplingParams, list[int], list[int]]
#                    ↑    ↑               ↑           ↑
#                  index  params     prompt_ids  output_ids (reference)
```

### 9. BatchUpdate Creation and Dispatch

**File**: `vllm/v1/sample/logits_processor/state.py` (lines 116-142)

When `refresh_metadata()` is called on the batch:

```python
def refresh_metadata(self):
    """Refresh batch metadata and update logits processors."""

    # Generate batch update from accumulated changes
    batch_update = self.batch_update_builder.get_and_reset(self.num_reqs)

    # Update all logits processors
    for logit_proc in self.logitsprocs.all:
        logit_proc.update_state(batch_update)  # <-- CALLED ON EACH PROCESSOR

    if batch_update:
        self.sampling_metadata = self._make_sampling_metadata()
```

**File**: `vllm/v1/sample/logits_processor/state.py` (lines 39-58)

```python
class BatchUpdateBuilder:
    """Builder for BatchUpdate objects."""

    def __init__(self):
        self._removed: list[int] = []
        self.moved: list[tuple[int, int, MoveDirectionality]] = []
        self.added: list[tuple[int, SamplingParams, list[int], list[int]]] = []
        self.batch_changed: bool = False

    def get_and_reset(self, batch_size: int) -> Optional[BatchUpdate]:
        """Create BatchUpdate and reset builder state."""

        if not any((self._removed, self.moved, self.added)):
            return None

        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=self._removed,
            moved=self.moved,
            added=self._added,  # <-- Contains (index, SamplingParams, ...)
        )

        # Reset state
        self._removed = []
        self.moved = []
        self.added = []

        return batch_update
```

### 10. BatchUpdate Interface

**File**: `vllm/v1/sample/logits_processor/interface.py` (lines 36-56)

```python
# Type aliases for batch update components
RemovedRequest = int  # Batch index of removed request
AddedRequest = tuple[int, SamplingParams, list[int], list[int]]
#                    ↑    ↑               ↑           ↑
#                  index  params     prompt_ids  output_ids
MovedRequest = tuple[int, int, MoveDirectionality]

@dataclass(frozen=True)
class BatchUpdate:
    """Batch state change information for logits processors.

    Contains information about requests added, removed, or moved
    within the batch since the last update.
    """
    batch_size: int
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]      # <-- List of (index, SamplingParams, ...)
    moved: Sequence[MovedRequest]
```

### 11. AdapterLogitsProcessor.update_state()

**File**: `vllm/v1/sample/logits_processor/__init__.py` (lines 262-267)

```python
class AdapterLogitsProcessor(LogitsProcessor):
    """Wrapper for per-request logits processors."""

    def __init__(self, vllm_config, device, is_pin_memory):
        # Map req index -> logits processor state (partial function)
        self.req_info: dict[int, partial[torch.Tensor]] = {}

    def update_state(self, batch_update: Optional[BatchUpdate]):
        """Update processor state based on batch changes."""
        process_dict_updates(
            self.req_info,
            batch_update,
            self._new_state,  # Callback to create per-request processor
        )

    def _new_state(
        self,
        params: SamplingParams,  # <-- RECEIVES SAMPLING PARAMS
        prompt_ids: list[int],
        output_ids: list[int],
    ) -> Optional[partial[torch.Tensor]]:
        """Create state for a new request."""

        # Call subclass implementation
        if req_lp := self.new_req_logits_processor(params):
            # Determine arguments based on signature
            args = [prompt_ids, output_ids] if (len(
                inspect.signature(req_lp).parameters) == 3) else [output_ids]
            # Return partial with pre-filled arguments
            return partial(req_lp, *args)
        return None

    @abstractmethod
    def new_req_logits_processor(
        self,
        params: SamplingParams,  # <-- SUBCLASS RECEIVES THIS
    ) -> Optional[RequestLogitsProcessor]:
        """Create per-request logits processor from sampling params.

        Subclasses implement this to extract relevant parameters
        and return a processor, or None if not applicable.
        """
        raise NotImplementedError
```

### 12. process_dict_updates Utility

**File**: `vllm/v1/sample/logits_processor/builtin.py` (lines 235-273)

```python
def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: Optional[BatchUpdate],
    new_state: Callable[[SamplingParams, list[int], list[int]], Optional[T]]
) -> bool:
    """Update dict-based logits processor state.

    Args:
        req_entries: Dict mapping batch index -> processor state
        batch_update: Batch changes to process
        new_state: Callback to create state for new requests

    Returns:
        True if any changes were made
    """
    if not batch_update:
        return False

    updated = False

    # Process added requests
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        # Call new_state callback with SamplingParams
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    # Process removed requests
    for index in batch_update.removed:
        if req_entries.pop(index, None) is not None:
            updated = True

    # Process moved requests (batch compaction)
    for index1, index2, directionality in batch_update.moved:
        if directionality == MoveDirectionality.SAME:
            continue

        state = req_entries.pop(index1, None)
        if state is not None:
            req_entries[index2] = state
            updated = True

    return updated
```

### 13. ReplayLogitsProcessor Implementation

**File**: `vllm/v1/sample/logits_processor/__init__.py` (lines 307-357)

```python
class ReplayLogitsProcessor(AdapterLogitsProcessor):
    """Forces sampling to follow a replay token sequence when provided.

    Usage: Set SamplingParams.extra_args with:
        - "replay_token_ids": List of token IDs to replay
        - "replay_force_stop_token_id": (Optional) Token to force after sequence
    """

    TOKEN_IDS_KEY = "replay_token_ids"
    FORCE_STOP_TOKEN_KEY = "replay_force_stop_token_id"

    def is_argmax_invariant(self) -> bool:
        # Must modify logits before argmax, so not invariant
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,  # <-- RECEIVES PARAMS FROM BATCH UPDATE
    ) -> Optional[RequestLogitsProcessor]:
        """Create replay processor if replay tokens are provided."""

        # Check for extra_args
        if not params.extra_args:
            return None

        # Extract replay token IDs
        replay_tokens = params.extra_args.get(self.TOKEN_IDS_KEY)
        if replay_tokens is None:
            return None

        # Validate token sequence
        if not isinstance(replay_tokens, Sequence):
            logger.warning(
                "Expected %s to be a sequence of token ids, got %s instead.",
                self.TOKEN_IDS_KEY, type(replay_tokens))
            return None

        # Convert to int list
        try:
            token_ids = [int(token) for token in replay_tokens]
        except (TypeError, ValueError):
            logger.warning("Unable to coerce %s into token ids; "
                           "ignoring replay logits processor.",
                           self.TOKEN_IDS_KEY)
            return None

        if not token_ids:
            logger.warning("Replay token list is empty; skipping replay mode.")
            return None

        # Extract optional force stop token
        force_stop_token: Optional[int] = None
        if (stop_token := params.extra_args.get(self.FORCE_STOP_TOKEN_KEY)
            ) is not None:
            try:
                force_stop_token = int(stop_token)
            except (TypeError, ValueError):
                logger.warning("Unable to coerce %s (%s) into int; "
                               "ignoring replay stop token.",
                               self.FORCE_STOP_TOKEN_KEY, stop_token)

        # Create and return processor
        return _ReplayRequestProcessor(token_ids, force_stop_token)
```

### 14. Per-Request Processor

**File**: `vllm/v1/sample/logits_processor/__init__.py` (lines 282-305)

```python
class _ReplayRequestProcessor:
    """Per-request processor that forces the next token to a replay target.

    This is the actual callable that gets invoked during sampling.
    """

    def __init__(self, token_ids: list[int],
                 force_stop_token_id: Optional[int]) -> None:
        self._token_ids = token_ids
        self._force_stop_token_id = force_stop_token_id

    def __call__(self, output_ids: list[int],
                 logits: torch.Tensor) -> torch.Tensor:
        """Modify logits to force next token to replay target."""

        # Determine which token to force based on current position
        next_index = len(output_ids)
        force_token: Optional[int] = None

        if next_index < len(self._token_ids):
            # Force next token from replay sequence
            force_token = self._token_ids[next_index]
        elif self._force_stop_token_id is not None:
            # Force stop token after sequence completes
            force_token = self._force_stop_token_id

        if force_token is None:
            # No forcing needed
            return logits

        # Set all logits to -inf except target token
        logits.fill_(float("-inf"))
        logits[force_token] = 0.0
        return logits
```

### 15. Application During Sampling

**File**: `vllm/v1/sample/logits_processor/__init__.py` (lines 269-279)

```python
def apply(self, logits: torch.Tensor) -> torch.Tensor:
    """Apply per-request processors to batch logits."""

    if self.req_info:
        # Apply per-request logits processors to corresponding rows
        for req_idx, req_lp in self.req_info.items():
            req_logits = logits[req_idx]
            new_logits = req_lp(req_logits)  # <-- Calls _ReplayRequestProcessor
            if new_logits is not req_logits:
                # Modify logits tensor row in-place if necessary
                logits[req_idx] = new_logits

    return logits
```

---

## Usage Example

To use `ReplayLogitsProcessor` for replay mode:

```python
from vllm import AsyncLLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

# Initialize AsyncLLM with ReplayLogitsProcessor enabled (it's built-in)
engine_args = AsyncEngineArgs(model="your-model")
async_llm = AsyncLLM.from_engine_args(engine_args)

# Create SamplingParams with replay configuration
sampling_params = SamplingParams(
    temperature=0.0,  # Temperature doesn't matter since we force tokens
    max_tokens=100,
    extra_args={
        "replay_token_ids": [1234, 5678, 9101, 1112],  # Tokens to replay
        "replay_force_stop_token_id": 2,  # Optional: force EOS after sequence
    }
)

# Generate with replay mode
request_id = "unique_request_id"
async for output in async_llm.generate(
    prompt="Your prompt here",
    sampling_params=sampling_params,
    request_id=request_id,
):
    print(f"Generated tokens: {output.outputs[0].token_ids}")
    # Output will be exactly [1234, 5678, 9101, 1112, 2]
```

---

## Key Files Reference

1. **Entry Point**: `vllm/v1/engine/async_llm.py:270-310`
2. **Processor**: `vllm/v1/engine/processor.py:314-437`
3. **EngineCoreRequest**: `vllm/v1/engine/__init__.py:43-71`
4. **Request Wrapper**: `vllm/v1/request.py:122-144`
5. **Scheduler**: `vllm/v1/core/sched/scheduler.py:1094-1095, 568-571`
6. **SchedulerOutput**: `vllm/v1/core/sched/output.py:25-56, 118-162`
7. **GPU Model Runner**: `vllm/v1/worker/gpu_model_runner.py:534-668`
8. **InputBatch** (CRITICAL): `vllm/v1/worker/gpu_input_batch.py:274-293`
9. **BatchUpdateBuilder**: `vllm/v1/sample/logits_processor/state.py:39-58, 116-142`
10. **BatchUpdate Interface**: `vllm/v1/sample/logits_processor/interface.py:36-56`
11. **AdapterLogitsProcessor**: `vllm/v1/sample/logits_processor/__init__.py:180-280`
12. **ReplayLogitsProcessor**: `vllm/v1/sample/logits_processor/__init__.py:282-357`
13. **Utility Function**: `vllm/v1/sample/logits_processor/builtin.py:235-273`

---

## Critical Insights

### 1. SamplingParams is Preserved Throughout Pipeline
The `SamplingParams` object (or its clone) flows through the entire system without modification:
- Cloned in `Processor.process_inputs()`
- Stored in `EngineCoreRequest`
- Transferred to `Request`
- Passed to `NewRequestData`
- Stored in `CachedRequestState`
- **Registered in `BatchUpdateBuilder.added`**

### 2. The Critical Handoff Point
**File**: `vllm/v1/worker/gpu_input_batch.py:289-291`

This is where `SamplingParams` enters the logits processor update pipeline:
```python
self.batch_update_builder.added.append(
    (new_req_index, request.sampling_params,
     request.prompt_token_ids, request.output_token_ids)
)
```

### 3. BatchUpdate as the Communication Mechanism
`BatchUpdate` serves as the communication protocol between the batch management layer and the logits processor layer:
- `added`: New requests with their `SamplingParams`
- `removed`: Requests that finished or were aborted
- `moved`: Requests that changed position during batch compaction

### 4. AdapterLogitsProcessor Pattern
The `AdapterLogitsProcessor` provides a clean abstraction for per-request processors:
- **`update_state()`**: Handles batch updates generically
- **`_new_state()`**: Creates processor partials with pre-filled arguments
- **`new_req_logits_processor()`**: Subclass implements param extraction
- **`apply()`**: Applies processors to batch logits

### 5. Output Token List Reference
The `output_token_ids` list reference is passed to the processor, allowing it to:
- Track generation progress (position in replay sequence)
- Make stateful decisions based on already-generated tokens
- Avoid needing separate state management

### 6. ReplayLogitsProcessor Configuration
Use `SamplingParams.extra_args` to pass custom configuration:
```python
extra_args = {
    "replay_token_ids": [token1, token2, ...],  # Required
    "replay_force_stop_token_id": eos_token_id,  # Optional
}
```

The `extra_args` dictionary is the extension point for custom logits processors.

---

## Sequence Diagram

```
User Code
    |
    | SamplingParams(extra_args={"replay_token_ids": [...]})
    ↓
AsyncLLM.add_request()
    |
    ↓
Processor.process_inputs()
    |
    | params.clone()
    ↓
EngineCoreRequest(sampling_params=...)
    |
    ↓
EngineCore → Scheduler → SchedulerOutput
    |
    ↓
GPUModelRunner._update_states()
    |
    | CachedRequestState(sampling_params=..., output_token_ids=[])
    ↓
InputBatch.add_request()
    |
    | _register_add_request()
    | BatchUpdateBuilder.added.append((idx, params, prompt_ids, output_ids))
    ↓
InputBatch.refresh_metadata()
    |
    | batch_update = BatchUpdateBuilder.get_and_reset()
    ↓
for each LogitsProcessor:
    |
    | processor.update_state(batch_update)
    ↓
ReplayLogitsProcessor.update_state()
    |
    | process_dict_updates(req_info, batch_update, _new_state)
    |
    | for (idx, params, prompt_ids, output_ids) in batch_update.added:
    |     state = _new_state(params, prompt_ids, output_ids)
    |     req_info[idx] = state
    ↓
_new_state(params, prompt_ids, output_ids)
    |
    | req_lp = new_req_logits_processor(params)
    | return partial(req_lp, output_ids)
    ↓
new_req_logits_processor(params)
    |
    | replay_tokens = params.extra_args.get("replay_token_ids")
    | force_stop = params.extra_args.get("replay_force_stop_token_id")
    | return _ReplayRequestProcessor(replay_tokens, force_stop)
    ↓
req_info[idx] = partial(_ReplayRequestProcessor(...), output_ids)

... Later during sampling ...

Sampler.apply_logits_processors()
    |
    ↓
ReplayLogitsProcessor.apply(logits)
    |
    | for idx, req_lp in req_info.items():
    |     logits[idx] = req_lp(logits[idx])  # Calls _ReplayRequestProcessor
    ↓
_ReplayRequestProcessor.__call__(output_ids, logits)
    |
    | next_index = len(output_ids)
    | force_token = token_ids[next_index]
    | logits.fill_(-inf)
    | logits[force_token] = 0.0
    | return logits
    ↓
Forced token sampled → Added to output_ids
```

---

## Summary

The data flow from `SamplingParams` to `ReplayLogitsProcessor` is:

1. **User creates** `SamplingParams(extra_args={"replay_token_ids": [...]})`
2. **AsyncLLM** passes params to `Processor`
3. **Processor** clones params and embeds in `EngineCoreRequest`
4. **Scheduler** propagates params through `Request` → `NewRequestData` → `SchedulerOutput`
5. **GPUModelRunner** creates `CachedRequestState` with params and output list
6. **InputBatch** registers request in `BatchUpdateBuilder.added` **(CRITICAL STEP)**
7. **BatchUpdate** is created with `added=[(idx, params, prompt_ids, output_ids)]`
8. **LogitsProcessor.update_state()** is called with `BatchUpdate`
9. **process_dict_updates()** iterates over `added` tuples
10. **ReplayLogitsProcessor.new_req_logits_processor()** extracts replay tokens from `params.extra_args`
11. **_ReplayRequestProcessor** is created and stored in `req_info[idx]`
12. **During sampling**, `apply()` calls the processor to force tokens

The key insight is that **`SamplingParams` is preserved throughout the entire pipeline** and **`BatchUpdateBuilder.added` is the bridge** between request management and logits processing.
