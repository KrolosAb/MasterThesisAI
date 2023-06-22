import random
import numpy as np
import networkx as nx

def random_node_sampling(G, num_samples):
    """This function selects a random subset of nodes from the graph"""

    nodes = list(G.nodes()) # Get all nodes in the graph
    sampled_nodes = random.sample(nodes, num_samples) # Randomly select num_samples nodes
    
    return G.subgraph(sampled_nodes)

def node_type_sampling(G, num_samples, type_weights={'uri': 1, 'literal': 1, 'predicate': 1}):
    """This function selects nodes based on their type"""

    nodes_by_type = {'uri': [], 'literal': [], 'predicate': []}
    for node, data in G.nodes(data=True):
        nodes_by_type[data['node_type']].append(node) # Categorizing nodes based on their type

    sampled_nodes = []
    for node_type, weight in type_weights.items():
        type_nodes = nodes_by_type[node_type] # Getting nodes of a specific type
        num_samples_type = int(num_samples * weight) # Determining the number of samples to select for that type

        if len(type_nodes) < num_samples_type:
            num_samples_type = len(type_nodes) # Adjusting the number of samples if there are fewer nodes of this type

        sampled_nodes.extend(np.random.choice(type_nodes, num_samples_type, replace=False)) # Randomly selecting nodes from this type

    return G.subgraph(sampled_nodes)

def edge_type_sampling(G, num_samples, type_weights ={'subj_pred': 1, 'pred_obj': 1}):
    """This function selects nodes based on the types of edges they are connected to"""

    nodes_by_edge_type = {'subj_pred': [], 'pred_obj': []}
    for u, v, data in G.edges(data=True):
        edge_type = data['edge_type']
        nodes_by_edge_type[edge_type].append(u) # Categorizing nodes based on the edge types they are connected to
        nodes_by_edge_type[edge_type].append(v)

    sampled_nodes = []
    for edge_type, weight in type_weights.items():
        type_nodes = list(set(nodes_by_edge_type[edge_type])) # Getting nodes of a specific edge type
        num_samples_type = int(num_samples * weight) # Determining the number of samples to select for this type

        if len(type_nodes) < num_samples_type:
            num_samples_type = len(type_nodes) # Adjusting the number of samples if there are fewer nodes of this type

        sampled_nodes.extend(np.random.choice(type_nodes, num_samples_type, replace=False)) # Randomly selecting nodes from this type

    return G.subgraph(sampled_nodes)

def degree_based_sampling(G, num_samples):
    """This function selects nodes based on their degree (the number of edges connected to them)"""

    degrees = [d for n, d in G.degree()] # Getting the degree for each node in the graph
    probabilities = [d / sum(degrees) for d in degrees] # Calculating the probability of selecting each node based on its degree
    nodes = list(G.nodes())
    sampled_nodes = np.random.choice(nodes, size=num_samples, p=probabilities, replace=False) # Randomly selecting nodes based on their probabilities
    return G.subgraph(sampled_nodes)

def degree_centrality_sampling(G, num_samples):
    """This function selects nodes based on their degree centrality"""

    centrality = nx.degree_centrality(G) # Calculating the degree centrality for each node in the graph
    probabilities = [centrality[node] for node in G.nodes()] # Getting the degree centrality values as probabilities
    total = sum(probabilities)
    probabilities = [p/total for p in probabilities] # Normalizing the probabilities
    nodes = list(G.nodes())
    sampled_nodes = np.random.choice(nodes, size=num_samples, p=probabilities, replace=False) # Randomly selecting nodes based on their probabilities
    return G.subgraph(sampled_nodes)

def pagerank_sampling(G, num_samples, alpha=0.8):
    """This function selects nodes based on their PageRank"""
    
    pagerank = nx.pagerank(G, alpha=alpha) # Calculating the PageRank for each node in the graph
    probabilities = [pagerank[node] for node in G.nodes()] # Getting the PageRank values as probabilities
    nodes = list(G.nodes())
    sampled_nodes = np.random.choice(nodes, size=num_samples, p=probabilities, replace=False) # Randomly selecting nodes based on their probabilities
    return G.subgraph(sampled_nodes)

def node_edge_type_sampling(G, num_samples, type_weights=[{'uri': 1, 'literal': 1, 'predicate': 1}, {'subj_pred': 1, 'pred_obj': 1}, 0.8]):
    """This function combines node type and edge type sampling"""

    num_samples = int(num_samples * 0.5)

    # Applying node type sampling and degree based sampling independently
    G_node_type = node_type_sampling(G, num_samples, type_weights[0])
    G_edge_type = edge_type_sampling(G, num_samples, type_weights[1])

    # Combining the sampled nodes from each method
    sampled_nodes = list(G_node_type.nodes()) + list(G_edge_type.nodes())

    return G.subgraph(sampled_nodes)

def node_type_pagerank_sampling(G, num_samples, type_weights=[{'uri': 1, 'literal': 1, 'predicate': 1}, {'subj_pred': 1, 'pred_obj': 1}, 0.8]):
    """This function combines node type and pagerank sampling"""

    num_samples = int(num_samples * 0.5)

    # Applying node type sampling and degree based sampling independently
    G_node_type = node_type_sampling(G, num_samples, type_weights[0])
    G_pagerank = pagerank_sampling(G, num_samples, type_weights[2])

    # Combining the sampled nodes from each method
    sampled_nodes = list(G_node_type.nodes()) + list(G_pagerank.nodes())

    return G.subgraph(sampled_nodes)

def edge_type_pagerank_sampling(G, num_samples, type_weights=[{'uri': 1, 'literal': 1, 'predicate': 1}, {'subj_pred': 1, 'pred_obj': 1}, 0.8]):
    """This function combines edge type and pagerank sampling"""
  
    num_samples = int(num_samples * 0.5)

    # Applying node type sampling and degree based sampling independently
    G_edge_type = edge_type_sampling(G, num_samples, type_weights[1])
    G_pagerank = pagerank_sampling(G, num_samples, type_weights[2])

    # Combining the sampled nodes from each method
    sampled_nodes = list(G_edge_type.nodes()) + list(G_pagerank.nodes())

    return G.subgraph(sampled_nodes)
    
def node_type_degree_based_sampling(G, num_samples, type_weights=[{'uri': 1, 'literal': 1, 'predicate': 1}, {'subj_pred': 1, 'pred_obj': 1}, 0.8]):
    """This function combines node type and degree based sampling"""

    num_samples = int(num_samples * 0.5)

    # Applying node type sampling and degree based sampling independently
    G_node_type = node_type_sampling(G, num_samples, type_weights[0])
    G_degree_based = degree_based_sampling(G, num_samples)

    # Combining the sampled nodes from each method
    sampled_nodes = list(G_node_type.nodes()) + list(G_degree_based.nodes())

    return G.subgraph(sampled_nodes)
