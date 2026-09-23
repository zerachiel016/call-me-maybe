import contextlib

from llm_sdk import Small_LLM_Model
from src.models import FunctionDef, Prompt
import numpy as np

model = Small_LLM_Model()

def get_context(prompt: Prompt, functions: list[FunctionDef]) -> str:
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

def encode_ids(txt):
    return model.encode(txt).tolist()[0]

def masked(logits, target_ids):
    log_array = np.array(logits)
    masked = np.full(log_array.shape, -np.inf, dtype=log_array.dtype)
    masked[target_ids] = log_array[target_ids]
    return np.argmax(masked)

def get_fn_name(context, fn_names):
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
        record += model.decode(t_max)
        context_ids.append(t_max)
    return record


def get_number(context) -> str:
    context_ids = list(context)
    target_ids = encode_ids("0123456789")
    val = ""

    while True:
        logits = model.get_logits_from_input_ids(context_ids)
        val_frag_id = np.argmax(logits)
        if val_frag_id not in target_ids:
            break
        val += model.decode(val_frag_id)
        context_ids.append(val_frag_id)

    return val

def get_string(context) -> str:
    context_ids = list(context)
    out = ""

    while True:
        logits = model.get_logits_from_input_ids(context_ids)
        out_frag_id = np.argmax(logits)
        out_frag = model.decode(out_frag_id)
        context_ids.append(out_frag_id)
        if '"' in out_frag:
            out_frag = out_frag[:out_frag.find('"')]
            out += out_frag
            break
        out += out_frag
    return out



def decoder(prompts, functions) -> list[dict]:
    res = []
    for prompt in prompts:
        _data = {}
        context = get_context(prompt, functions)
        fn_name = get_fn_name(encode_ids(context), [f.name for f in functions])
        context += fn_name + '", "parameters": {' 
        _data["prompt"] = prompt.prompt
        _data["name"] = fn_name

        param_pairs = {}
        function = next(fn for fn in functions if fn.name == fn_name) 
        for i, (k, v) in enumerate(function.parameters.items()):
            context += f'"{k}": ' if not i else f', "{k}": '
            match v['type']:
                case "number":
                    param_pairs[k] = float(get_number(encode_ids(context)))
                    context += str(param_pairs[k])
                case "string":
                    context += '"'
                    param_pairs[k] = get_string(encode_ids(context))
                    context += param_pairs[k] + '"'
                case "integer":
                    param_pairs[k] = get_number(encode_ids(context))
                    context += param_pairs[k]
                case "boolean":
                    param_pairs[k] = get_fn_name(encode_ids(context), ["True", "False"])
                    context += param_pairs[k]
        _data["parameters"] = param_pairs
        res.append(_data)
    return res
