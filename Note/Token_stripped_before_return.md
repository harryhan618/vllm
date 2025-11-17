# Token Stripping in vLLM v1 Engine

## Overview
This document details all tokens that are stripped from the output string before returning to users in vLLM v1 engine (v0.10.2). While these tokens are removed from the text output, they are preserved in the `token_ids` array for debugging and analysis purposes.

## Types of Tokens Stripped

### 1. EOS (End-of-Sequence) Token
- **When stripped**: When `stop_terminated=True` and `include_stop_str_in_output=False` (default)
- **Location**: `vllm/v1/engine/detokenizer.py:104-110`
- **Mechanism**: The last token is excluded from detokenization when identified as EOS

### 2. Stop Tokens
- **Source**: Token IDs specified in `stop_token_ids` parameter
- **Detection**: `vllm/v1/core/sched/utils.py:66-69`
- **Behavior**: Same stripping mechanism as EOS token

### 3. Special Tokens
- **Default**: `skip_special_tokens=True` (line 175 in `sampling_params.py`)
- **Scope**: All tokens in `tokenizer.all_special_tokens`
- **Implementation**: `vllm/transformers_utils/detokenizer_utils.py:33-39`

#### Common Special Tokens Stripped:
- **Universal**: `<s>` (BOS), `</s>` (EOS), `<unk>`, `<pad>`
- **Llama models**: `<s>`, `</s>`, `<unk>`
- **GPT models**: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`
- **Mistral models**: `[INST]`, `[/INST]`, `<<SYS>>`, `<</SYS>>`
- **Model-specific**: Custom special tokens defined by the model

### 4. Stop Strings
- **Source**: Text patterns specified in `stop` parameter list
- **Behavior**: Output text is truncated at the stop string match
- **Location**: `vllm/v1/engine/detokenizer.py:129-141`

### 5. Spaces Between Special Tokens
- **Condition**: When `spaces_between_special_tokens=False` and special tokens are adjacent
- **Effect**: Suppresses spaces that would normally be inserted between special tokens
- **Location**: `vllm/v1/engine/detokenizer.py:199-226`

## Key Control Parameters

```python
class SamplingParams:
    # Core stripping parameters with defaults
    skip_special_tokens: bool = True          # Strip special tokens from output
    spaces_between_special_tokens: bool = True # Add spaces between special tokens
    include_stop_str_in_output: bool = False  # Include stop strings/tokens in output
    ignore_eos: bool = False                  # Whether to ignore EOS token for generation
    stop: Optional[list[str]] = None          # Stop strings that trigger completion
    stop_token_ids: Optional[list[int]] = None # Stop token IDs that trigger completion
```

## Token Stripping Flow

### 1. Generation Phase (`check_stop()` in `utils.py`)
```python
def check_stop(request, max_model_len, pooler_output=None):
    # Check for EOS token
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    # Check for stop tokens
    if last_token_id in sampling_params.stop_token_ids:
        request.status = RequestStatus.FINISHED_STOPPED
        return True
```

### 2. Detokenization Phase (`detokenizer.py`)
```python
def update(self, new_token_ids, stop_terminated):
    if stop_terminated and not self.include_stop_str_in_output:
        # Strip the stop token from text generation
        skipped_stop_token_id = new_token_ids[-1]
        new_token_ids = new_token_ids[:-1]  # Remove from text conversion

    # Detokenize remaining tokens
    for new_token_id in new_token_ids:
        self.token_ids.append(new_token_id)
        self.output_text += self.decode_next(new_token_id)
```

### 3. Special Token Filtering (`detokenizer_utils.py`)
```python
all_special_tokens = set(tokenizer.all_special_tokens) if skip_special_tokens else ()
for token in output_tokens:
    if token in all_special_tokens:
        continue  # Skip this token in text output
```

## Request Completion Pipeline

```
EngineCore.step()
    ↓
Scheduler.check_stop() → Detects EOS/stop conditions
    ↓
Detokenizer.update() → Strips stop/special tokens
    ↓
OutputProcessor.make_request_output() → Assembles final output
    ↓
RequestOutput → Returns to user (text without stripped tokens, token_ids with all tokens)
```

## How to Control Token Stripping

### To See All Tokens in Output
```python
sampling_params = SamplingParams(
    skip_special_tokens=False,        # Show special tokens
    include_stop_str_in_output=True,  # Show stop strings/tokens
    spaces_between_special_tokens=True # Maintain spaces
)
```

### To Access Raw Token IDs
```python
# Tokens are preserved in the token_ids field even when stripped from text
output = llm.generate(prompt, sampling_params)
token_ids = output.token_ids  # Contains all tokens including stripped ones
text = output.text            # Text with tokens stripped according to parameters
```

## Important Notes

1. **Token Preservation**: Stripped tokens are removed from `output_text` but preserved in `token_ids` array
2. **Default Behavior**: By default, special tokens and stop tokens are stripped (`skip_special_tokens=True`, `include_stop_str_in_output=False`)
3. **Finish Reasons**: The `finish_reason` field indicates why generation stopped (STOP, LENGTH, ABORT)
4. **Stop vs Continue**: `ignore_eos=True` prevents EOS from stopping generation but doesn't affect stripping if EOS appears

## File References

- **Stop Detection**: `/vllm/v1/core/sched/utils.py:43-69`
- **Token Stripping**: `/vllm/v1/engine/detokenizer.py:91-142`
- **Special Token Handling**: `/vllm/transformers_utils/detokenizer_utils.py:15-51`
- **Parameters**: `/vllm/sampling_params.py:96-219`
- **Output Assembly**: `/vllm/v1/engine/output_processor.py:180-287`