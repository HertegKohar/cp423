"""
Author: Kelvin Kellner
"""
import re


def compute_soundex(term):
    """
    Compute the soundex code for a given term
    -----
    Args:
        term (str): Term to compute soundex code for
    Returns:
        soundex (str): Soundex code for 'term'
    """
    soundex = ""
    soundex += term[0].upper()
    for char in term[1:].lower():
        if char in "bfpv":
            soundex += "1"
        elif char in "cgjkqsxz":
            soundex += "2"
        elif char in "dt":
            soundex += "3"
        elif char in "l":
            soundex += "4"
        elif char in "mn":
            soundex += "5"
        elif char in "r":
            soundex += "6"
        else:
            soundex += "0"
    soundex = re.sub(r"(.)\1+", r"\1", soundex)
    soundex = re.sub(r"0", "", soundex)
    soundex = soundex[:4].ljust(4, "0")
    return soundex