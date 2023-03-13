"""
Author: Herteg Kohar
"""
import argparse
import ast
import math


def unary_encode(n):
    return "1" * n + "0"


def binary_encode(n, width):
    r = ""
    for i in range(width):
        if ((1 << i) & n) > 0:
            r = "1" + r
        else:
            r = "0" + r
    return r


def gamma_encode(n):
    logn = int(math.log(n, 2))
    return unary_encode(logn) + " " + binary_encode(n, logn)


def delta_encode(n):
    logn = int(math.log(n, 2))
    if n == 1:
        return "0"
    loglog = int(math.log(logn + 1, 2))
    residual = logn + 1 - int(math.pow(2, loglog))
    return (
        unary_encode(loglog)
        + " "
        + binary_encode(residual, loglog)
        + " "
        + binary_encode(n, logn)
    )


def gamma_decode(n):
    n = str(n)
    i = 0
    while n[i] == "1":
        i += 1
    return int(math.pow(2, i) + int(n[i + 1 :], 2))


def delta_decode(n):
    n = str(n)
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
