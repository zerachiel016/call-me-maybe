from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Literal
import json

class FunctionDef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = Field(min_length=1) 
    parameters: dict[str, dict[Literal["type"], Literal["string", "number", "boolean", "integer"]]]
    returns: dict[Literal["type"], Literal["string", "number", "boolean", "integer"]]

    @field_validator("name")
    @classmethod
    def must_be_identifier(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(f"{v!r} is not a valid Python identifier")
        return v

class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key: {key!r}")
        result[key] = value
    return result

def parse_definition(path):
    with open(path, "r", encoding="utf-8") as handle:  
        data = json.load(handle, object_pairs_hook=reject_duplicates)
    return [FunctionDef.model_validate(d) for d in data]

def parse_prompts(path):
    with open(path, "r", encoding="utf-8") as handle:  
        data = json.load(handle, object_pairs_hook=reject_duplicates)
    return [Prompt.model_validate(d) for d in data]
