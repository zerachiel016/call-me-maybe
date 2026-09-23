import json
import sys
import os
import argparse
from src.models import parse_definition, parse_prompts
from src.decoder import decoder
from json import dump


def main():
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
    os.makedirs(path, exist_ok=True)

    with open(path + args.output[len(path):], "w") as o:
        res = decoder(parse_prompts(args.input), parse_definition(args.functions_definition))
        dump(res, o, indent=4)

if __name__ == "__main__":
    sys.exit(main())
