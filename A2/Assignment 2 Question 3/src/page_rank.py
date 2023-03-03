import argparse
import ast

"""
Author: Bryan Gadd
"""

"""
Notes/thoughts:
 - Only loop through the txt file once
 - Store and update the nodes (store id, PR, nodes it points to, etc.)
 - For each iteration is just re-loops through the existing stored nodes
"""
class Node:
    nodeId: int # id of the node
    nodesPointTo: list[int] = [] #nodes that this node points to
    nodesPointFrom: list[int] = [] # nodes that point to this node

    def __init__(self, id):
        self.nodeId = id

    # adds nodeId to list of nodes that this node is linked to (points to)
    def add_linked_node(self, nodeId):
        if (nodeId not in self.nodesPointTo):
            self.nodesPointTo.append(nodeId)
    
    # adds nodeId to list of nodes that this node is linked to (points to)
    def add_connected_node(self, nodeId):
        if (nodeId not in self.nodesPointFrom):
            self.nodesPointFrom.append(nodeId)


def page_rank(maxiteration, lambda_, thr, nodes):
    nodeList:list[Node] = []

    # test output to make sure variables are good
    print("maxiteration= " + str(maxiteration) + ", lambda_= " + str(lambda_) + ", thr= " + str(thr) + ", nodes= " + str(nodes))

    file = open("Web-Stanford.txt", "r")
    line = file.readline()

    # get to actual nodes
    while (line[0].isdigit() == False):
        line = file.readline()

    # until end of file
    while (line != ""):
        # extract node number and the node it points to
        extractedString = line.split()
        nodeId = int(extractedString[0])
        linkedNode = int(extractedString[1])

        # verifying I can get the numbers
        print(str(nodeId) + " --> " + str(linkedNode))

        #add first node to list
        if (nodeList.count == 0):
            node = Node(nodeId)
            node.add_linked_node(linkedNode)
            nodeList.append(node)

        else:
            #if node was already added then just add to linked nodes
            if (nodeId in nodeList):
                node = Node(nodeList.index(nodeId))
                node.add_linked_node(linkedNode)
                #add to connected list of node this one links to
            
            #add new node
            else:
                node = Node(nodeId)
                node.add_linked_node(linkedNode)
                nodeList.append(node)
                
        # next line
        line = file.readline()

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
