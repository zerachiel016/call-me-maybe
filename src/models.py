"""Pydantic models and JSON parsing utilities.

This module defines the data models used to validate and represent
function definitions and user prompts loaded from JSON input files.
It also provides parsing functions with duplicate-key detection.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Literal
import json


class FunctionDef(BaseModel):
    """Pydantic model representing a callable function definition.

    Validates and stores a function's metadata as loaded from the
    functions_definition.json input file, including its name,
    description, parameter types, and return type.

    Attributes:
        name: The function name, must be a valid Python identifier.
        description: A non-empty human-readable description of the function.
        parameters: A mapping of parameter names to their type descriptors.
            Each value is a dict with a single ``"type"`` key whose value
            is one of ``"string"``, ``"number"``, ``"boolean"``, or
            ``"integer"``.
        returns: A dict with a single ``"type"`` key describing the
            return type, following the same type literals as parameters.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = Field(min_length=1)
    parameters: dict[
        str,
        dict[
            Literal["type"],
            Literal["string", "number", "boolean", "integer"],
        ],
    ]
    returns: dict[
        Literal["type"],
        Literal["string", "number", "boolean", "integer"],
    ]

    @field_validator("name")
    @classmethod
    def must_be_identifier(cls, v: str) -> str:
        """Validate that the function name is a valid Python identifier.

        Args:
            v: The function name string to validate.

        Returns:
            The validated function name, unchanged.

        Raises:
            ValueError: If ``v`` is not a valid Python identifier.
        """
        if not v.isidentifier():
            raise ValueError(f"{v!r} is not a valid Python identifier")
        return v

    @field_validator("description")
    @classmethod
    def must_not_be_whitespace(cls, v: str) -> str:
        """Validate that the description is not solely whitespace.

        Args:
            v: The description string to validate.

        Returns:
            The validated description, unchanged.

        Raises:
            ValueError: If ``v`` contains only whitespace.
        """
        if v.isspace():
            raise ValueError(f"{v!r} is an empty string")
        return v


class Prompt(BaseModel):
    """Pydantic model representing a user prompt for function calling.

    Validates and stores a single natural language prompt as loaded
    from the function_calling_tests.json input file.

    Attributes:
        prompt: A non-empty string containing the user's natural
            language request.
    """

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def must_not_be_whitespace(cls, v: str) -> str:
        """Validate that the prompt is not solely whitespace.

        Args:
            v: The prompt string to validate.

        Returns:
            The validated prompt, unchanged.

        Raises:
            ValueError: If ``v`` contains only whitespace.
        """
        if v.isspace():
            raise ValueError(f"{v!r} is an empty string")
        return v


def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a dict from key-value pairs, raising on duplicate keys.

    Intended for use as the ``object_pairs_hook`` argument to
    ``json.load`` so that duplicate keys in JSON objects are detected
    and rejected rather than silently overwritten.

    Args:
        pairs: A list of ``(key, value)`` tuples as provided by the
            JSON decoder.

    Returns:
        A dictionary built from the provided pairs.

    Raises:
        ValueError: If any key appears more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key: {key!r}")
        result[key] = value
    return result


def parse_definition(path: str) -> list[FunctionDef]:
    """Parse a JSON file of function definitions into validated models.

    Reads the specified JSON file, checks for duplicate keys, and
    validates each entry against the ``FunctionDef`` schema.

    Args:
        path: Filesystem path to the functions_definition JSON file.

    Returns:
        A list of validated ``FunctionDef`` instances.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If duplicate keys are found in any JSON object.
        pydantic.ValidationError: If any entry fails schema validation.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=reject_duplicates)
    return [FunctionDef.model_validate(d) for d in data]


def parse_prompts(path: str) -> list[Prompt]:
    """Parse a JSON file of prompts into validated models.

    Reads the specified JSON file, checks for duplicate keys, and
    validates each entry against the ``Prompt`` schema.

    Args:
        path: Filesystem path to the function_calling_tests JSON file.

    Returns:
        A list of validated ``Prompt`` instances.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If duplicate keys are found in any JSON object.
        pydantic.ValidationError: If any entry fails schema validation.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=reject_duplicates)
    return [Prompt.model_validate(d) for d in data]
