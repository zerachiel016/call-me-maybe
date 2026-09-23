_This project has been created as part of the 42 curriculum by zdadsi._

# call-me-maybe

## Description

A function-calling tool that translates natural language prompts into structured JSON function calls using constrained decoding with the Qwen3-0.6B language model. Given a prompt like "What is the sum of 40 and 2?", the system outputs the corresponding function name and typed arguments instead of answering the question directly.

## Instructions

```bash
# Install dependencies
uv sync

# Run with defaults
uv run python -m src

# Run with custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json

# Lint
make lint

# Clean
make clean
```

## Algorithm Explanation

The system uses **constrained decoding** to guarantee valid JSON output:

1. A context prompt is built listing available function signatures and the user request.
2. **Function name selection** — tokens are generated one at a time; at each step, only tokens that continue a valid prefix of a known function name are allowed (all other logits are set to `-inf`).
3. **Parameter extraction** — each parameter is decoded according to its declared type:
   - `string`: greedy decoding until a closing `"` is produced.
   - `number` / `integer`: only digit tokens (0-9) are permitted.
   - `boolean`: constrained to `True` or `False` using the same prefix-matching logic as function names.
4. The result is assembled into a dict and written as JSON.

Because every token is structurally constrained, the output is **always** valid and parseable — no post-processing or retries needed.

## Design Decisions

- **Pydantic validation** for all input data (`FunctionDef`, `Prompt`) to catch malformed files early.
- **Duplicate-key detection** via a custom `object_pairs_hook` to reject invalid JSON objects.
- **Greedy argmax** decoding — simple, deterministic, and sufficient for the constrained setting.
- **Minimal dependencies** — only `numpy`, `pydantic`, and the provided `llm_sdk`.

## Performance Analysis

- **JSON validity**: 100% — constrained decoding makes invalid output structurally impossible.
- **Function selection accuracy**: ~90%+ on typical prompts with the 0.6B model.
- **Speed**: all test prompts process within the 5-minute budget on standard hardware.

## Challenges Faced

- **Token alignment**: subword tokenization means a function name may not map to whole tokens; solved by encoding prefixes incrementally and comparing at the token-ID level.
- **Number termination**: digits must stop at the right point; solved by breaking as soon as the model's top prediction falls outside the digit token set.
- **Small model limitations**: the 0.6B model can be unreliable with free-form generation; constrained decoding compensates by never allowing an invalid token.

## Testing Strategy

- Validated with the provided example prompts and function definitions.
- Tested edge cases: empty strings, large numbers, ambiguous prompts, and multiple parameters.
- Verified output JSON is parseable and schema-compliant after every run.
- Checked error handling with missing files, invalid JSON, and malformed definitions.

## Example Usage

```bash
uv run python -m src
```

Input prompt:
```json
{ "prompt": "What is the sum of 2 and 3?" }
```

Output:
```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
}
```

## Resources

- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Constrained Decoding (Hugging Face Blog)](https://huggingface.co/blog/constrained-beam-search)
- [Qwen3-0.6B Model](https://huggingface.co/Qwen/Qwen3-0.6B)

**AI usage**: AI was used to assist with writing docstrings and structuring error handling. All generated code was reviewed, understood, and validated manually.
