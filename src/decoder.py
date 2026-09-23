"""Constrained decoding engine for translating prompts into function calls.

This module implements a constrained decoding pipeline that uses a
small language model to convert natural language prompts into structured
JSON function calls. It guides the model's token generation step by
step, masking invalid tokens to guarantee well-formed output.
"""

from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore
from src.models import FunctionDef, Prompt
import numpy as np

model = Small_LLM_Model()


def get_context(prompt: Prompt, functions: list[FunctionDef]) -> str:
    """Build the LLM context string for a given prompt and function set.

    Constructs a multi-line prompt that instructs the model to act as a
    function-calling assistant, lists the available function signatures,
    includes the user request, and begins the JSON output template so
    that constrained decoding can continue from there.

    Args:
        prompt: The user prompt to process.
        functions: The list of available function definitions.

    Returns:
        A formatted context string ready to be tokenised and fed
        to the model.
    """
    context = ["you are a function calling assitant",
               "you have to choose the right function for the user request",
               "and output that in a json format",
               "allowed functions:"]
    signatures = [
        f'{f.name}({f.parameters}): {f.description}'
        for f in functions
    ]
    context.extend(signatures)
    context.append(f'the user request: {prompt.prompt}')
    context.append("now the answer :")
    context.append('JSON``` {"name": "')
    return "\n".join(context)


def encode_ids(txt: str) -> list[int]:
    """Encode a text string into a list of token IDs.

    Wraps the model's tokeniser to convert raw text into the integer
    token IDs that the LLM expects as input.

    Args:
        txt: The text string to tokenise.

    Returns:
        A list of integer token IDs representing the input text.
    """
    return [int(x) for x in model.encode(txt).tolist()[0]]


def masked(logits: list[float], target_ids: list[int]) -> int:
    """Select the highest-probability token from a constrained set.

    Applies constrained decoding by setting all logits outside the
    allowed ``target_ids`` to negative infinity, then returns the
    index of the remaining maximum logit.

    Args:
        logits: The raw logit scores produced by the model for every
            token in the vocabulary.
        target_ids: Indices of the tokens that are valid choices at
            this decoding step.

    Returns:
        The token ID (vocabulary index) with the highest logit among
        the allowed tokens.
    """
    log_array = np.array(logits)
    masked = np.full(log_array.shape, -np.inf, dtype=log_array.dtype)
    masked[target_ids] = log_array[target_ids]
    return int(np.argmax(masked))


def get_fn_name(context: list[int], fn_names: list[str]) -> str:
    """Decode a function name using constrained token-by-token generation.

    Uses the model to generate tokens one at a time, restricting each
    step to tokens that continue a valid prefix of one of the allowed
    function names. Generation stops when the accumulated string
    exactly matches one of the candidates.

    Args:
        context: The current sequence of token IDs (the encoded
            context so far).
        fn_names: The list of valid function name strings to choose
            from.

    Returns:
        The function name selected by the model.
    """
    context_ids = list(context)
    record = ""
    while record not in fn_names:
        logits = model.get_logits_from_input_ids(context_ids)
        target_ids = [
            encode_ids(fn_name)[len(encode_ids(record))]
            for fn_name in fn_names
            if fn_name.startswith(record)
        ]

        t_max = masked(logits, target_ids)
        record += str(model.decode([t_max]))
        context_ids.append(t_max)
    return record


def get_number(context: list[int]) -> str:
    """Decode a numeric string using constrained generation.

    Generates digits one token at a time, restricting each step to
    digit tokens (0-9). Stops when the model's top prediction is a
    non-digit token.

    Args:
        context: The current sequence of token IDs (the encoded
            context so far).

    Returns:
        A string of digit characters representing the decoded number.
    """
    context_ids = list(context)
    target_ids = encode_ids("0123456789")
    val = ""

    while True:
        logits = model.get_logits_from_input_ids(context_ids)
        val_frag_id = int(np.argmax(logits))
        if val_frag_id not in target_ids:
            break
        val += str(model.decode([val_frag_id]))
        context_ids.append(val_frag_id)

    return val


def get_string(context: list[int]) -> str:
    """Decode a free-form string value using unconstrained generation.

    Generates tokens one at a time using greedy decoding (highest
    logit) until a double-quote character is encountered, which
    signals the end of the JSON string value.

    Args:
        context: The current sequence of token IDs (the encoded
            context so far, including the opening quote).

    Returns:
        The decoded string content (without surrounding quotes).
    """
    context_ids = list(context)
    out = ""

    while True:
        logits = model.get_logits_from_input_ids(context_ids)
        out_frag_id = int(np.argmax(logits))
        out_frag = str(model.decode([out_frag_id]))
        context_ids.append(out_frag_id)
        if '"' in out_frag:
            out_frag = out_frag[:out_frag.find('"')]
            out += out_frag
            break
        out += out_frag
    return out


def decoder(prompts: list[Prompt],
            functions: list[FunctionDef]) -> list[dict[str, Any]]:
    """Translate a list of prompts into structured function call dicts.

    For each prompt, this function uses constrained decoding to:
    1. Select the most appropriate function name from the available
       definitions.
    2. Extract each parameter value with type-appropriate decoding
       (number, string, integer, or boolean).

    Args:
        prompts: The list of user prompts to process.
        functions: The list of available function definitions.

    Returns:
        A list of dictionaries, each containing:
        - ``"prompt"``: the original prompt text,
        - ``"name"``: the selected function name,
        - ``"parameters"``: a dict of parameter name-value pairs with
          values cast to the types specified in the function definition.
    """
    res: list[dict[str, Any]] = []
    for prompt in prompts:
        _data: dict[str, Any] = {}
        context = get_context(prompt, functions)
        fn_name = get_fn_name(
            encode_ids(context), [f.name for f in functions]
        )
        context += fn_name + '", "parameters": {'
        _data["prompt"] = prompt.prompt
        _data["name"] = fn_name

        param_pairs: dict[str, Any] = {}
        function = next(
            fn for fn in functions if fn.name == fn_name
        )
        for i, (k, v) in enumerate(function.parameters.items()):
            context += f'"{k}": ' if not i else f', "{k}": '
            match v['type']:
                case "number":
                    param_pairs[k] = float(
                        get_number(encode_ids(context))
                    )
                    context += str(param_pairs[k])
                case "string":
                    context += '"'
                    s_val = get_string(
                        encode_ids(context)
                    )
                    param_pairs[k] = s_val
                    context += s_val + '"'
                case "integer":
                    param_pairs[k] = int(
                        get_number(encode_ids(context))
                    )
                    context += str(param_pairs[k])
                case "boolean":
                    raw = get_fn_name(
                        encode_ids(context),
                        ["True", "False"],
                    )
                    param_pairs[k] = raw == "True"
                    context += raw
        _data["parameters"] = param_pairs
        res.append(_data)
    return res
