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
    nodesPointTo: list[int] #nodes that this node points to
    nodesPointFrom: list[int] # nodes that point to this node

    def __init__(self, id):
        self.nodeId = id
        self.nodesPointTo = []
        self.nodesPointFrom = []

    # adds nodeId to list of nodes that this node is linked to (points to)
    def add_linked_node(self, nodeId):
        if (nodeId not in self.nodesPointTo):
            self.nodesPointTo.append(nodeId)
    
    # adds nodeId to list of nodes that this node is linked to (points to)
    def add_connected_node(self, nodeId):
        if (nodeId not in self.nodesPointFrom):
            self.nodesPointFrom.append(nodeId)

#determine if node is in list
def node_exists(nodeList:list[Node], nodeId):
        for node in nodeList:
            if (node.nodeId == nodeId):
                return True
        return False

#find index
def node_in_list(nodeList:list[Node], nodeId):
        for node in nodeList:
            if (node.nodeId == nodeId):
                return node
        return None

def page_rank(maxiteration, lambda_, thr, nodes):
    nodeList:list[Node] = []

    # test output to make sure variables are good
    #print("maxiteration= " + str(maxiteration) + ", lambda_= " + str(lambda_) + ", thr= " + str(thr) + ", nodes= " + str(nodes))

    file = open("test.txt", "r")
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
        #print(str(nodeId) + " --> " + str(linkedNode))

        #add first node to list
        if (nodeList.count == 0):
            node = Node(nodeId)
            node.add_linked_node(linkedNode)
            nodeList.append(node)

        else:
            #if node was already added then just add to linked nodes
            if (node_exists(nodeList, nodeId)):
                print("Node '" + str(nodeId) +"' exists. Adding '" + str(linkedNode) + "' to it's list...")
                node = node_in_list(nodeList, nodeId)
                node.add_linked_node(linkedNode)
                #add to connected list of node this one links to
            
            #add new node
            else:
                print("Node '" + str(nodeId) +"' doesn't exist. Adding it to the list with '" + str(linkedNode) + "' for the first linked node")
                node = Node(nodeId)
                node.add_linked_node(linkedNode)
                nodeList.append(node)
        
        print("Node '" + str(node.nodeId) + "' nodesPointTo is now: " + str(node.nodesPointTo))
        # next line
        line = file.readline()

    file.close()

    for node in nodeList:
        print("Id= " + str(node.nodeId))
        print("nodesPointTo: " + str(node.nodesPointTo))
        # for linkedNode in node.nodesPointTo:
        #     print("Linked Node= " + str(linkedNode))
        # for connectedNodes in node.nodesPointFrom:
        #     print("Connected Node= " + str(connectedNodes))
        

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
