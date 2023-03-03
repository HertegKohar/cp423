import argparse
import ast


def gamma_encode(n):
    """Encode a non-negative integer n using gamma coding."""
    if n < 0:
        return "ERROR"
    if n == 0:
        return "0"
    # calculate the binary representation of n
    binary = bin(n)[2:]
    # calculate the unary representation of the length of the binary string
    unary = "1" * (len(binary) - 1)
    # combine the unary and binary representations
    return unary + binary


def gamma_decode(code):
    """Decode a string of bits in gamma coding to a non-negative integer."""
    # find the length of the unary part (number of consecutive 1s)
    if len(code) == 0 or "0" not in code or "1" not in code:
        return "ERROR"
    length = 0
    while code[length] == "1":
        length += 1
    # extract the binary part
    binary = code[length:]
    # calculate the decimal value of the binary part
    n = int("1" + binary, 2)
    return n


def delta_encode(n):
    """Encode a non-negative integer n using delta coding."""
    if n < 0:
        return "ERROR"
    if n == 0:
        return "0"
    # calculate the binary representation of n
    binary = bin(n)[2:]
    # calculate the unary representation of the length of the binary string
    unary = gamma_encode(len(binary))
    # combine the unary and binary representations
    return unary + binary


def delta_decode(code):
    """Decode a string of bits in delta coding to a non-negative integer."""
    # find the length of the unary part (number of consecutive 1s)
    if len(code) == 0 or "0" not in code or "1" not in code:
        return "ERROR"
    length = 0
    while code[length] == "1":
        length += 1
    # extract the binary part
    binary = code[length + 1 : 2 * length + 1]
    # calculate the decimal value of the binary part
    n = int("1" + binary, 2)
    # add the remaining binary digits to the value
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
    parser.add_argument("numbers", type=str, help="Numbers to encode/decode")
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
