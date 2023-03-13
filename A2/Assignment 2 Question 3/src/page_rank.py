import argparse

"""
Author: 
- Bryan Gadd
- Herteg Kohar
"""



class Node:
    # nodeId: int # id of the node
    # nodesPointTo: set[int] #nodes that this node points to
    # nodesPointFrom: set[int] # nodes that point to this node
    # pageRank: float

    def __init__(self, id):
        self.nodeId = id
        self.nodesPointTo = set()
        self.nodesPointFrom = set()
        self.pageRank = 0

    # adds nodeId to list of nodes that this node is linked to (points to)
    def add_linked_node(self, nodeId):
        self.nodesPointTo.add(nodeId)

    # adds nodeId to list of nodes that this node is linked to (points to)
    def add_connected_node(self, nodeId):
        self.nodesPointFrom.add(nodeId)


nodeList: dict[int, Node] = {}


def page_rank(maxiteration, lambda_, thr, nodes):
    totalNumNodes = len(nodeList)

    # set up initial values and formula's
    initialPageRank = 1 / totalNumNodes
    calc = (lambda_ / totalNumNodes) + (1 - lambda_)
    previousPageRanks = {}

    # set initial Pagerank for each node
    for node in nodeList:
        nodeList[node].pageRank = initialPageRank

    for i in range(maxiteration):
        # store previous pageranks for calculation
        for n in nodeList:
            previousPageRanks[n] = nodeList[n].pageRank

        # calculate PageRank for each node
        for node in nodeList:
            # print("Calculating PR for node '" + str(node) + "'")
            sum = 0

            for j in nodeList[node].nodesPointFrom:

                # add to the sum
                sum += previousPageRanks[j] / len(nodeList[j].nodesPointTo)
                # print("Sum is now= " + str(sum))

            # calculate pagerank for the node
            nodeList[node].pageRank = calc * sum

        # should this be checked for entire list (vector is how it's normally done)
        errSum = 0
        for n in nodeList:
            errSum += (nodeList[n].pageRank - previousPageRanks[n]) ** 2
        if (errSum) < thr:
            print("Convergence detected")
            break
        else:
            print("Convergence not detected")
        # print("Node '" + str(node) + "' pagerank is now= " + str(nodeList[node].pageRank))

    # print the nodes specified at launch
    print("\nPageRank of requested Nodes:")
    for node in nodes:
        nodePagerank = nodeList[int(node)].pageRank
        print("Node (" + str(node) + ") pagerank= " + str(nodePagerank))

    # TODO- Debug: Without threshold the total is 0.003... (way too small) | With treashold is 1.0000000000028555 (to much)
    totalPagerank = 0
    for node in nodeList:
        totalPagerank += nodeList[node].pageRank
    print("Total Pagerank= " + str(totalPagerank))


""""
Add's the connected node to the list or updates if it already exists
"""


def connected_node(nodeId, linkedNode):
    if linkedNode in nodeList:
        nodeList[linkedNode].add_connected_node(nodeId)
    # if node doesn't exist yet we need to add it + update connected list
    else:
        node = Node(linkedNode)
        node.add_connected_node(nodeId)
        nodeList[linkedNode] = node
    return


def graph_retrieval():

    # read file (Web-Stanford.txt)
    with open("Web-Stanford.txt") as f:
        for line in f:
            # skips lines that aren't nodes
            if line[0].isdigit() == False:
                continue

            # extract node number and the node it points to
            extractedString = line.split()
            nodeId = int(extractedString[0])
            linkedNode = int(extractedString[1])

            # add first node to list
            if len(nodeList) == 0:
                node = Node(nodeId)
                node.add_linked_node(linkedNode)
                nodeList[nodeId] = node

                # Update or add the the connected node
                connected_node(nodeId, linkedNode)

            # not first node
            else:
                # if node already exists then just add to linked nodes
                if nodeId in nodeList:
                    # print("Node '" + str(nodeId) +"' exists. Adding '" + str(linkedNode) + "' to it's list...")
                    node = nodeList[nodeId]
                    node.add_linked_node(linkedNode)

                    # Update or add the the connected node
                    connected_node(nodeId, linkedNode)

                # add new node
                else:
                    # print("Node '" + str(nodeId) +"' doesn't exist. Adding it to the list with '" + str(linkedNode) + "' for the first linked node")
                    node = Node(nodeId)
                    node.add_linked_node(linkedNode)
                    nodeList[nodeId] = node

                    # Update or add the the connected node
                    connected_node(nodeId, linkedNode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command Line Interface for PageRank")
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
        dest="lambda_",
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

    # TODO- Should be able to do this in a single or 2 steps...
    sptringEdit = args.nodes.replace("[", "")
    sptringEdit = sptringEdit.replace("]", "")
    nodes: list = sptringEdit.split(",")

    # test output to make sure variables are good
    # print("maxiteration= " + str(args.maxiteration) + ", lambda_= " + str(args.lambda_) + ", thr= " + str(args.thr) + ", nodes= " + str(nodes))

    # parse the graph
    graph_retrieval()

    # calculate page rank + output selected nodes
    page_rank(args.maxiteration, args.lambda_, args.thr, nodes)
