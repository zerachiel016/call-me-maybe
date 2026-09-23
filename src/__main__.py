"""Entry point for the function-calling CLI application.

Parses command-line arguments for input/output file paths, loads
function definitions and user prompts, runs constrained decoding,
and writes the resulting function calls to a JSON output file.
"""

import sys
import os
import argparse
from src.models import parse_definition, parse_prompts
from src.decoder import decoder
from json import dump


def main() -> None:
    """Run the function-calling pipeline from the command line.

    Parses ``--input``, ``--output``, and ``--functions_definition``
    arguments (with sensible defaults), reads and validates the input
    files, runs the constrained-decoding pipeline, and writes the
    results as indented JSON to the output path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json")
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json")
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json")
    args = parser.parse_args()

    path = os.path.dirname(args.output)
    if path:
        os.makedirs(path, exist_ok=True)

    prompts = parse_prompts(args.input)
    functions = parse_definition(args.functions_definition)
    with open(args.output, "w", encoding="utf-8") as o:
        res = decoder(prompts, functions)
        dump(res, o, indent=4)


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
