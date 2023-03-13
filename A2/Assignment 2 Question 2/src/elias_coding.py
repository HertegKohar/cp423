"""
Author: Herteg Kohar
"""
import argparse
import ast
import math
import re

valid_binary = re.compile(r"^[01]+$")


def unary_encode(n):
    """Creates a unary encoded string of length n.

    Args:
        n (int): The length of the unary encoded string.

    Returns:
        str: The unary encoded string.
    """
    return "1" * n + "0"


def binary_encode(n, width):
    """Creates a binary encoded string of length width.

    Args:
        n (int): The number to be encoded.
        width (int): The length of the binary encoded string.

    Returns:
        str: The binary encoded string.
    """
    r = ""
    for i in range(width):
        if ((1 << i) & n) > 0:
            r = "1" + r
        else:
            r = "0" + r
    return r


def gamma_decode_leading_0s(n):
    """Decodes a gamma encoded string with leading zeros.

    Args:
        n (str): The gamma encoded string.

    Returns:
        int: The decoded number.
    """
    n = n.lstrip("0")  # remove leading zeros
    i = len(n)
    return int(math.pow(2, i) + int(n[1:], 2))


def gamma_encode(n):
    """Encodes a number using gamma encoding."""
    if n < 1:
        return "ERROR"
    logn = int(math.log(n, 2))
    return unary_encode(logn) + binary_encode(n, logn)


def delta_decode_leading_0s(n):
    """Decodes a delta encoded string with leading zeros."""
    if not valid_binary.match(n):
        return "ERROR"
    n = n.lstrip("0")  # remove leading zeros
    i = len(n)
    j = i + 1
    while n[j] == "0":
        j += 1
    loglog = int(math.pow(2, i) + int(n[i + 1 : j], 2))
    residual = int(n[j : j + loglog], 2)
    logn = int(math.pow(2, loglog) + residual)
    return int(math.pow(2, logn) + int(n[j + loglog :], 2))


def delta_encode(n):
    """Encodes a number using delta encoding."""
    if n < 1:
        return "ERROR"
    logn = int(math.log(n, 2))
    if n == 1:
        return "0"
    loglog = int(math.log(logn + 1, 2))
    residual = logn + 1 - int(math.pow(2, loglog))
    return (
        unary_encode(loglog) + binary_encode(residual, loglog) + binary_encode(n, logn)
    )


def gamma_decode(n):
    """Decode a gamma encoded string."""
    if not valid_binary.match(n):
        return "ERROR"
    try:
        if n.startswith("1"):
            n = str(n)
            i = 0
            while n[i] == "1":
                i += 1
            return int(math.pow(2, i) + int(n[i + 1 :], 2))
        else:
            return gamma_decode_leading_0s(n)
    except Exception as e:
        return "ERROR"


def delta_decode(n):
    """Decode a delta encoded string."""
    if not valid_binary.match(n):
        return "ERROR"
    try:
        n = str(n)
        if n.startswith("1"):
            i = 0
            while n[i] == "1":
                i += 1
            j = i + 1
            while n[j] == "0":
                j += 1
            loglog = int(math.pow(2, i) + int(n[i + 1 : j], 2))
            residual = int(n[j : j + loglog], 2)
            logn = int(math.pow(2, loglog) + residual)
            return int(math.pow(2, logn) + int(n[j + loglog :], 2))
        else:
            return delta_decode_leading_0s(n)
    except Exception as e:
        return "ERROR"


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
        "[" + ",".join(f"'{x}'" for x in args.numbers.strip("[]").split(",")) + "]"
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
