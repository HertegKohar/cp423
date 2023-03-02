import argparse
import ast

"""
Author: Bryan Gadd
"""

def page_rank(maxiteration, lambda_, thr, nodes):
    #test output to make sure variables are good
    print("maxiteration= " + str(maxiteration) + ", lambda_= " + str(lambda_) + ", thr= " + str(thr) + ", nodes= " + str(nodes))
    
    file = open("Web-Stanford.txt", "r")

    #ignore any lines that start with # (few before you reach the actual nodes).

    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command Line Interface for Wikipedia Processing"
    )
    parser.add_argument(
        "--maxiteration",
        help="Maximum number of iterations",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--lambda",
        help="Lambda",
        type=float,
        required=True,
        dest='lambda_',
    )
    parser.add_argument(
        "--thr",
        help="Threshold",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--nodes",
        help="NodeIDs that are retrieved at the end with their page ranks",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    page_rank(args.maxiteration, args.lambda_, args.thr, args.nodes)
