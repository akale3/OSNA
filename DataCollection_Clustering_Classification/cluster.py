"""
cluster.py
"""
import datetime
import json

import matplotlib.pyplot as plt
import networkx as nx


def find_jaccard_similarity(userFriends, otherUserFriends):
    commonUserList = set(userFriends).intersection(otherUserFriends)
    unionList = set().union(userFriends, otherUserFriends)
    return len(commonUserList) / len(unionList)


def create_graph(users):
    G = nx.Graph()
    for user in users:
        G.add_node(user['screen_name'])
        for otherUser in users:
            if otherUser != user:
                jaccardSimilarity = find_jaccard_similarity(user['friend_ids'], otherUser['friend_ids'])
                if jaccardSimilarity > 0.005:
                    G.add_edge(user['screen_name'], otherUser['screen_name'])
    return G


def draw_network(graph, filename):
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(graph, pos,
                           graph.nodes(),
                           node_size=50)
    nx.draw_networkx_edges(graph, pos,
                           graph.edges(),
                           edge_color='gray')

    plt.axis('off')
    plt.savefig(filename)


def readUserFile(filePath):
    userFriendList = list()
    f = open(filePath, 'r')
    for line in f:
        if len(line) > 0:
            userFriendList.append(json.loads(line))
    return userFriendList


def find_best_edge(graph):
    eb = nx.edge_betweenness_centrality(graph)
    return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]


result = []


def girvan_newman(G):
    if G.order() in range(2, 5):
        result.append(G.nodes())
        return

    if G.order() <= 1:
        return

    components = [c for c in nx.connected_component_subgraphs(G)]
    while len(components) == 1:
        edge_to_remove = find_best_edge(G)
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]

    for c in components:
            girvan_newman(c)
    return


def get_subgraph(graph, min_degree):
    subGraph = nx.Graph()
    for node in graph.nodes():
        if len(graph.neighbors(node)) >= min_degree:
            subGraph.add_node(node)
    for outerNode in subGraph.nodes():
        for innerNode in subGraph.nodes():
            if graph.has_edge(innerNode, outerNode):
                subGraph.add_edge(innerNode, outerNode)
    return subGraph


def main():
    startTime = datetime.datetime.now()

    """Read Friends Data  from file"""
    filePath = 'data/iphoneUsersFriends.txt'
    userFriendList = readUserFile(filePath)

    graph = create_graph(userFriendList)
    subGraph = get_subgraph(graph, 2)

    draw_network(subGraph, 'network.png')
    girvan_newman(subGraph)
    clusters = result
    if clusters:
        with open('cluster.txt', 'w') as summaryFile:
            summaryFile.write('Cluster Results: ')
            summaryFile.write('\n \n')
            summaryFile.write('Number of users collected: ' + str(len(userFriendList)))
            summaryFile.write('\n')
            summaryFile.write('Number of communities discovered: ' + str(len(clusters)))
            summaryFile.write('\n')
            totalUsers = 0
            for userList in clusters:
                totalUsers = totalUsers + len(userList)
            if len(clusters) > 0:
                summaryFile.write('Average number of users per community: ' + str(totalUsers / len(clusters)))
            else:
                summaryFile.write('Average number of users per community: ' + str(0))
            summaryFile.write('\n \n')
    else:
        with open('cluster.txt', 'w') as summaryFile:
            summaryFile.write('Cluster Results: ')
            summaryFile.write('\n \n')
            summaryFile.write('Number of users collected: ' + str(len(userFriendList)))
            summaryFile.write('\n')
            summaryFile.write('Number of communities discovered: ' + str(0))
            summaryFile.write('\n')
            summaryFile.write('Average number of users per community: ' + str(0))
            
    endTime = datetime.datetime.now()
    print(endTime - startTime)


if __name__ == '__main__':
    main()
