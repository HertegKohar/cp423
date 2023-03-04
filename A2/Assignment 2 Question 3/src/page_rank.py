import argparse

"""
Author: Bryan Gadd
"""

"""
Clean-up needed:
- Remove type hinting on varaibles.
- Remove any un-needed code.
"""
class Node:
    nodeId: int # id of the node
    nodesPointTo: set[int] #nodes that this node points to
    nodesPointFrom: set[int] # nodes that point to this node

    def __init__(self, id):
        self.nodeId = id
        self.nodesPointTo = set()
        self.nodesPointFrom = set()

    # adds nodeId to list of nodes that this node is linked to (points to)
    def add_linked_node(self, nodeId):
        self.nodesPointTo.add(nodeId)
    
    # adds nodeId to list of nodes that this node is linked to (points to)
    def add_connected_node(self, nodeId):
        self.nodesPointFrom.add(nodeId)


def page_rank(maxiteration, lambda_, thr, nodes, nodeList):
    print("page_rank() called")

def graph_retrieval(maxiteration, lambda_, thr, nodes):
    nodeList:dict[int, Node] = {}

    # test output to make sure variables are good
    #print("maxiteration= " + str(maxiteration) + ", lambda_= " + str(lambda_) + ", thr= " + str(thr) + ", nodes= " + str(nodes))
    # read file
    with open("Web-Stanford.txt") as f:
        for line in f:
            # skips lines that aren't nodes
            if (line[0].isdigit() == False):
                continue

            # extract node number and the node it points to
            extractedString = line.split()
            nodeId = int(extractedString[0])
            linkedNode = int(extractedString[1])

            # verifying I can get the numbers
            print(str(nodeId) + " --> " + str(linkedNode))

            # add first node to list
            if (len(nodeList) == 0):
                node = Node(nodeId)
                node.add_linked_node(linkedNode)
                nodeList[nodeId] = node

                # Connected
                # if node exists then add to connected list
                if (linkedNode in nodeList):
                    nodeList[linkedNode].add_connected_node(nodeId)
                
                # if node doesn't exist yet we need to add it + update connected list
                else:
                    node = Node(linkedNode)
                    node.add_connected_node(nodeId)
                    nodeList[linkedNode] = node

            # not first node
            else:
                #if node already exists then just add to linked nodes
                if (nodeId in nodeList):
                    #print("Node '" + str(nodeId) +"' exists. Adding '" + str(linkedNode) + "' to it's list...")
                    node = nodeList[nodeId]
                    node.add_linked_node(linkedNode)
                    
                    # Connected
                    # if node exists then add to connected list
                    if (linkedNode in nodeList):
                        nodeList[linkedNode].add_connected_node(nodeId)
                    
                    # if node doesn't exist yet we need to add it + update connected list
                    else:
                        node = Node(linkedNode)
                        node.add_connected_node(nodeId)
                        nodeList[linkedNode] = node
                        
                # add new node
                else:
                    #print("Node '" + str(nodeId) +"' doesn't exist. Adding it to the list with '" + str(linkedNode) + "' for the first linked node")
                    node = Node(nodeId)
                    node.add_linked_node(linkedNode)
                    nodeList[nodeId] = node

                    # Connected
                    #if node exists then update connected list
                    if (linkedNode in nodeList):
                        nodeList[linkedNode].add_connected_node(nodeId)
                    
                    #if node doesn't exist yet we need to add it + update connected list
                    else:
                        node = Node(linkedNode)
                        node.add_connected_node(nodeId)
                        nodeList[linkedNode] = node
            
            #print("Node '" + str(node.nodeId) + "' nodesPointTo is now: " + str(node.nodesPointTo))

    for node in nodeList:
        print("Id= " + str(node))
        print("nodesPointTo: " + str(nodeList[node].nodesPointTo))
        print("nodesPointFrom: " + str(nodeList[node].nodesPointFrom))

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
    graph_retrieval(args.maxiteration, args.lambda_, args.thr, args.nodes)
