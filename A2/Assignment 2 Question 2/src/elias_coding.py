"""
Author: Herteg Kohar
"""
import argparse
import ast


def gamma_encode(n):
    if n < 0:
        return "ERROR"
    if n == 0:
        return "0"
    binary = bin(n)[2:]
    unary = "1" * (len(binary) - 1)
    return unary + binary


def gamma_decode(code):
    if len(code) == 0 or "0" not in code or "1" not in code:
        return "ERROR"
    length = 0
    while code[length] == "1":
        length += 1
    binary = code[length:]
    n = int("1" + binary, 2)
    return n


def delta_encode(n):
    if n < 0:
        return "ERROR"
    if n == 0:
        return "0"
    binary = bin(n)[2:]
    unary = gamma_encode(len(binary))
    return unary + binary


def delta_decode(code):
    if len(code) == 0 or "0" not in code or "1" not in code:
        return "ERROR"
    length = 0
    while code[length] == "1":
        length += 1
    binary = code[length + 1 : 2 * length + 1]
    n = int("1" + binary, 2)
    remaining_binary = code[2 * length + 1 :]
    if remaining_binary:
        n = (n << len(remaining_binary)) | int(remaining_binary, 2)
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elias Coding")
    parser.add_argument("--alg", type=str, required=True, help="Elias Coding Algorithm")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encode", action="store_true", help="Numbers to encode")
    group.add_argument("--decode", action="store_true", help="Numbers to decode")
    parser.add_argument(
        "numbers", type=str, help="Numbers to encode/decode (no spaces)"
    )
    args = parser.parse_args()
    numbers = (
        "["
        + ",".join(f"'{int(x):d}'" for x in args.numbers.strip("[]").split(","))
        + "]"
    )
    numbers = ast.literal_eval(numbers)
    if args.encode:
        numbers = [int(x) for x in numbers]
    if args.alg == "gamma":
        if args.encode:
            for n in numbers:
                print(f"{n}: {gamma_encode(n)}")
        else:
            for n in numbers:
                print(f"{n}: {gamma_decode(n)}")
    else:
        if args.encode:
            for n in numbers:
                print(f"{n}: {delta_encode(n)}")
        else:
            for n in numbers:
                print(f"{n}: {delta_decode(n)}")
