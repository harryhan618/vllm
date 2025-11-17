# vLLM v1 Engine: Prompt to Token Logic and GPU Execution Flow

## Overview
This document traces the complete flow of how text prompts are tokenized and processed in vLLM v0.10.2's v1 engine, from the initial API request to GPU execution.

## Table of Contents
1. [Synchronous LLMEngine Flow](#1-synchronous-llmengine-flow)
2. [Asynchronous AsyncLLM Flow (OpenAI API)](#2-asynchronous-asyncllm-flow-openai-api)
3. [Key Components](#3-key-components)
4. [Architecture Highlights](#4-architecture-highlights)

---

## 1. Synchronous LLMEngine Flow

### 1.1 Request Entry Point
**File**: `vllm/v1/engine/llm_engine.py`

When a request arrives at `LLMEngine.add_request()` (line 196-238):
```python
def add_request(
    self,
    request_id: str,
    prompt: PromptType,
    params: Union[SamplingParams, PoolingParams],
    ...
) -> None:
    # Process raw inputs into the request
    prompt_str, request = self.processor.process_inputs(
        request_id, prompt, params, arrival_time, lora_request,
        tokenization_kwargs, trace_headers, priority)

    # Add to output processor and engine core
    self.output_processor.add_request(request, prompt_str, None, 0)
    self.engine_core.add_request(request)
```

### 1.2 Tokenization Process

#### Processor (`vllm/v1/engine/processor.py`)
The `Processor.process_inputs()` method (line 314-399) orchestrates tokenization:
- Validates parameters
- Calls `InputPreprocessor.preprocess()` (line 365)

#### InputPreprocessor (`vllm/inputs/preprocess.py`)
The tokenization flow:
1. `preprocess()` (line 896) → Determines if encoder-decoder or decoder-only
2. `_process_decoder_only_prompt()` (line 843) → For standard models
3. `_prompt_to_llm_inputs()` (line 515) → Handles different prompt types
4. `_process_text()` (line 443) → For text prompts
5. **`_tokenize_prompt()`** (line 190-210) → **Actual tokenization happens here**
   ```python
   def _tokenize_prompt(
       self,
       prompt: str,
       lora_request: Optional[LoRARequest],
       tokenization_kwargs: Optional[dict[str, Any]] = None,
   ) -> list[int]:
       tokenizer = self.get_tokenizer_group()
       return tokenizer.encode(prompt=prompt,
                              lora_request=lora_request,
                              **tokenization_kwargs)
   ```

### 1.3 Flow to GPU Workers

#### EngineCore (`vllm/v1/engine/core.py`)
- Receives the tokenized request in `add_request()` (line 225)
- Adds to scheduler (line 251)
- During `step()`, calls `model_executor.execute_model()` (line 293)

#### GPU Worker (`vllm/v1/worker/gpu_worker.py`)
- `Worker.execute_model()` (line 425) calls `model_runner.execute_model()`

#### GPUModelRunner (`vllm/v1/worker/gpu_model_runner.py`)
The `execute_model()` method (line 2000) prepares and executes:

1. **Prepare inputs** (`_prepare_inputs()`, line 862):
   - Converts token IDs to tensors (line 919-922):
   ```python
   torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                      0,
                      torch.from_numpy(token_indices),
                      out=self.input_ids.cpu[:total_num_scheduled_tokens])
   ```
   - Copies to GPU (line 948-957):
   ```python
   self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)
   self.positions.copy_to_gpu(total_num_scheduled_tokens)
   ```

2. **Model forward pass** (line 2064-2070):
   ```python
   model_output = self.model(
       input_ids=input_ids,
       positions=positions,
       intermediate_tensors=intermediate_tensors,
       inputs_embeds=inputs_embeds,
       **model_kwargs,
   )
   ```

---

## 2. Asynchronous AsyncLLM Flow (OpenAI API)

### 2.1 API Entry Point
**File**: `vllm/entrypoints/openai/api_server.py`

When `VLLM_USE_V1` is enabled:
- Request arrives at `/v1/chat/completions` endpoint (line 668)
- Creates AsyncLLM instance (line 214-228)

### 2.2 Chat Completion Processing
**File**: `vllm/entrypoints/openai/serving_chat.py`

`OpenAIServingChat.create_chat_completion()` (line 161-341):
1. Prepares messages and prompts (line 224-241)
2. Calls engine client (line 306-313):
   ```python
   generator = self.engine_client.generate(
       engine_prompt,
       sampling_params,
       request_id,
       lora_request=lora_request,
       trace_headers=trace_headers,
       priority=request.priority,
   )
   ```

### 2.3 AsyncLLM Processing
**File**: `vllm/v1/engine/async_llm.py`

#### Request Flow
1. `generate()` method (line 415) - Main entry point
2. Calls `add_request()` (line 462-471)
3. **Tokenization happens in `add_request()`** (line 307-309):
   ```python
   # Convert Input --> Request (includes tokenization)
   prompt_str, request = self.processor.process_inputs(
       request_id, prompt, params, arrival_time, lora_request,
       tokenization_kwargs, trace_headers, priority, data_parallel_rank)
   ```
4. Sends to engine core (line 338-339):
   ```python
   await self.engine_core.add_request_async(request)
   ```

### 2.4 Inter-Process Communication
**File**: `vllm/v1/engine/core_client.py`

AsyncMPClient handles communication (line 759):
- `add_request_async()` (line 902-904):
  ```python
  async def add_request_async(self, request: EngineCoreRequest) -> None:
      request.client_index = self.client_index
      await self._send_input(EngineCoreRequestType.ADD, request)
  ```
- Uses ZMQ sockets for IPC with EngineCore process

### 2.5 EngineCore to GPU Execution
The flow continues the same as synchronous:
1. EngineCore receives request via ZMQ
2. Scheduler manages request queue
3. Model executor runs on GPU worker
4. Results flow back through ZMQ to AsyncLLM

---

## 3. Key Components

### Tokenization Components
| Component | File | Purpose |
|-----------|------|---------|
| `Processor` | `vllm/v1/engine/processor.py` | High-level request processing and validation |
| `InputPreprocessor` | `vllm/inputs/preprocess.py` | Handles different prompt types and tokenization |
| `TokenizerGroup` | `vllm/transformers_utils/tokenizer_group.py` | Manages tokenizer instances and encoding |

### Execution Components
| Component | File | Purpose |
|-----------|------|---------|
| `EngineCore` | `vllm/v1/engine/core.py` | Core scheduling and execution orchestration |
| `GPUWorker` | `vllm/v1/worker/gpu_worker.py` | GPU worker management |
| `GPUModelRunner` | `vllm/v1/worker/gpu_model_runner.py` | Prepares inputs and runs model forward pass |

### Communication Components
| Component | File | Purpose |
|-----------|------|---------|
| `EngineCoreClient` | `vllm/v1/engine/core_client.py` | Client interface for engine communication |
| `AsyncMPClient` | `vllm/v1/engine/core_client.py` | Async multiprocess client using ZMQ |

---

## 4. Architecture Highlights

### V1 Engine Design Principles
1. **Separation of Concerns**
   - Tokenization in main process (Processor/InputPreprocessor)
   - Model execution in separate EngineCore process
   - GPU operations isolated in worker processes

2. **Multi-Process Architecture**
   - Main API server process handles HTTP and tokenization
   - EngineCore process manages scheduling and coordination
   - Worker processes handle GPU execution

3. **Communication Layer**
   - ZMQ sockets for high-performance IPC
   - Msgpack for efficient serialization
   - Async-compatible throughout

4. **Performance Optimizations**
   - CUDA graphs for efficient GPU execution
   - Efficient memory management with KV cache
   - Specialized attention backends
   - Batch processing and request scheduling

### Data Flow Summary
```
Text Prompt (API Request)
    ↓
Processor.process_inputs() [Tokenization]
    ↓
Token IDs + Metadata (EngineCoreRequest)
    ↓
ZMQ Socket [IPC]
    ↓
EngineCore Scheduler
    ↓
GPUModelRunner [Tensor Preparation]
    ↓
GPU Memory [input_ids, positions, etc.]
    ↓
Model Forward Pass
    ↓
Generated Tokens
```

### Key Insights
- **Tokenization always happens in the main process** using the same `Processor` class for both sync and async modes
- **Token IDs are serialized** and sent via ZMQ to the EngineCore process
- **GPU tensor preparation** happens in the worker process just before model execution
- **V1 architecture** enables better scalability through process isolation and efficient IPC

---

*Generated from vLLM v0.10.2 source code analysis*