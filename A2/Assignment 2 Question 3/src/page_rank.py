import argparse

def page_rank(maxiteration, lamb, thr, nodes):
    #ignore any lines that start with # (few before you reach the actual nodes).

    #need to perform pagerank on nodes...
    open("Web-Stanford.txt", "r")

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
        help="The lambda",
        type=float,
        required=True,
        dest='lambda_',
        #metavar='LAMBDA',
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
        nargs='+',
        required=False,
    )
    args = parser.parse_args()
    #page_rank(args.maxiteration, args.lambda_, args.thr, args.nodes)
    
    #test values
    page_rank(2, 0.15, args.thr, args.nodes)