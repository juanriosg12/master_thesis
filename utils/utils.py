import numpy as np
import pandas as pd
import networkx as nx
from typing import List
import matplotlib.pyplot as plt


def visualize_causal_graph( adjacency_matrix: np.ndarray,
                               confounder_info: List = None,
                               title: str = "Causal Graph",
                               y_parent_indices: List[int]= None):
        

        G = nx.DiGraph()
        n = adjacency_matrix.shape[0]

        # Add nodes X variable
        nodes = [f'X{i}' for i in range(n)]
        G.add_nodes_from(nodes)

        # Add Y node
        G.add_node("Y")

        for i in range(n):
            for j in range(n):
                if adjacency_matrix[i, j]==1:
                    G.add_edge(nodes[i],nodes[j])

        if y_parent_indices is not None:
            for parent_idx in y_parent_indices:
                G.add_edge(f"X{parent_idx}",'Y')

        # Add confounder nodes if present 
        if confounder_info:
            for conf_id, affected_nodes in confounder_info:
                conf_node = f'U{conf_id}'
                G.add_node(conf_node)
                for node_idx in affected_nodes:
                    G.add_edge(conf_node, nodes[node_idx])

        # Create layout

        pos = nx.spring_layout(G, seed=42)

        # Draw the graph 

        plt.figure(figsize=(14,10))

        x_nodes = [n for n in G.nodes() if n.startswith('X')]
        y_nodes = [n for n in G.nodes() if n.startswith('Y')]
        latent_nodes = [n for n in G.nodes() if n.startswith('U')]

        nx.draw_networkx_nodes(G,pos,nodelist=x_nodes,
                               node_color='lightblue',
                               node_size=500,alpha=0.9)
        
        if y_nodes:
            nx.draw_networkx_nodes(G,pos,nodelist=y_nodes,
                               node_color='gold',
                               node_size=800,alpha=0.9)
            
        if latent_nodes:
            nx.draw_networkx_nodes(G,pos,nodelist=latent_nodes,
                               node_color='lightcoral',
                               node_size=500,alpha=0.9,
                               node_shape='s')
            
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               arrows=True, arrowsize=20,
                               arrowstyle='->',width=2)
            

        nx.draw_networkx_labels(G,pos,font_size=10)

        plt.title(title,fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        return plt


def visualize_comparison(true_adj, discovered_adj, features_names,
                         y_parent_indices=None, discovered_y_parents=None,
                         true_confounders=None, discovered_confounders=None,
                         title_preix="Causal Grap Comparison"):
    
    n_features = len(features_names)

    fig , (ax1, ax2) = plt.subplots(1, 2, figsize= (20,8))

    # True graph
    G_true = nx.DiGraph()
    G_true.add_nodes_from(features_names)

    for i in range(n_features):
        for j in range(n_features):
            if true_adj[i, j] == 1:
                G_true.add_edge(features_names[i], features_names[j])

    if true_confounders:
        for conf_id, affected_nodes in true_confounders:
            conf_node = f'U{conf_id}'
            G_true.add_node(conf_node)
            for node_idx in affected_nodes:
                if node_idx < len(features_names):
                    G_true.add_edge(conf_node,features_names[node_idx])

    pos_true = nx.spring_layout(G_true, seed=42, k=2, iterations=50)

    x_nodes = [n for n in G_true.nodes() if n.startswith('X')]
    y_nodes = [n for n in G_true.nodes() if n == 'Y']
    u_nodes = [n for n in G_true.nodes() if n.startswith('U')]

    nx.draw_networkx_nodes(G_true,pos_true,nodelist=x_nodes,
                               node_color='lightblue', node_size=900,
                               alpha=0.9, ax=ax1)

    if y_nodes:
        nx.draw_networkx_nodes(G_true,pos_true,nodelist=y_nodes,
                               node_color='gold', node_size=900,
                               alpha=0.9, ax=ax1)
        
    if u_nodes:
         nx.draw_networkx_nodes(G_true,pos_true,nodelist=u_nodes,
                               node_color='lightcoral', node_size=900,
                               alpha=0.9, ax=ax1)
    
    x_to_x_edges = [(u,v) for u,v in G_true.edges()
                    if u.startswith('X') and v.startswith('X')]
    
    x_to_y_edges = [(u,v) for u,v in G_true.edges()
                    if u.startswith('X') and v.startswith('Y')]
    
    u_to_x_edges = [(u,v) for u,v in G_true.edges()
                    if u.startswith('U') and v.startswith('X')]
    
    u_to_y_edges = [(u,v) for u,v in G_true.edges()
                    if u.startswith('U') and v.startswith('Y')]
    
    nx.draw_networkx_edges(G_true, pos_true, edgelist=x_to_x_edges,
                           edge_color='gray', arrows=True,
                           arrowsize=20, width=2, ax=ax1)
    
    if x_to_y_edges: 
        nx.draw_networkx_edges(G_true, pos_true, edgelist=x_to_y_edges,
                           edge_color='green', arrows=True,
                           arrowsize=25, width=3, ax=ax1,
                           style='solid',alpha=0.8)

    if u_to_x_edges: 
        nx.draw_networkx_edges(G_true, pos_true, edgelist=u_to_x_edges,
                           edge_color='red', arrows=True,
                           arrowsize=20, width=2, ax=ax1,
                           style='dashed', alpha=0.7)
        
    if u_to_y_edges:
        nx.draw_networkx_edges(G_true, pos_true, edgelist=u_to_y_edges,
                        edge_color='red', arrows=True,
                        arrowsize=20, width=2, ax=ax1,
                        style='dashed', alpha=0.7)
        
    nx.draw_networkx_labels(G_true,pos_true,font_size=10, ax=ax1)

    ax1.set_title(f'{title_preix}\nTrue Causal Graph')
    ax1.axis('off')


    # Discovered graph

    G_disc = nx.DiGraph()
    G_disc.add_nodes_from(features_names)

    for i in range(n_features):
        for j in range(n_features):
            if discovered_adj[i, j] != 0:
                G_disc.add_edge(features_names[i], features_names[j])

    if discovered_confounders:
        conf_count=0
        for item in discovered_confounders:
            if isinstance(item, tuple) and len(item) == 2:
                var1, var2 = item

                conf_node = f'U{conf_count}'
                G_disc.add_node(conf_node)
                G_disc.add_edge(conf_node,var1)
                G_disc.add_edge(conf_node,var2)
                conf_count += 1

    pos_disc = nx.spring_layout(G_disc, seed=42, k=2, iterations=50)

    x_nodes_disc = [n for n in G_disc.nodes() if n.startswith('X')]
    y_nodes_disc = [n for n in G_disc.nodes() if n == 'Y']
    u_nodes_disc = [n for n in G_disc.nodes() if n.startswith('U')]

    nx.draw_networkx_nodes(G_disc,pos_disc,nodelist=x_nodes_disc,
                               node_color='lightcoral', node_size=900,
                               alpha=0.9, ax=ax2)

    if y_nodes_disc:
        nx.draw_networkx_nodes(G_disc,pos_disc,nodelist=y_nodes_disc,
                               node_color='orange', node_size=900,
                               alpha=0.9, ax=ax2)
        
    if u_nodes_disc:
         nx.draw_networkx_nodes(G_disc,pos_disc,nodelist=u_nodes_disc,
                               node_color='purple', node_size=900,
                               alpha=0.9, ax=ax2, node_shape='s')
    
    x_to_x_correct = []
    x_to_x_incorrect = []
    x_to_y_correct = []
    x_to_y_incorrect = []
    u_to_any_edges = []

    for u, v in G_disc.edges():
        if u.startswith('U'):
            u_to_any_edges.append((u,v))
            continue

        u_idx = features_names.index(u)
        v_idx = features_names.index(v)
        is_correct = (discovered_adj[u_idx, v_idx] != 0 and
                      true_adj[u_idx, v_idx] != 0)
        
        if v == 'Y':
            if is_correct:
                x_to_y_correct.append((u,v))
            else:
                x_to_y_incorrect.append((u,v))
        else:
            if is_correct:
                x_to_x_correct.append((u,v))
            else:
                x_to_x_incorrect.append((u,v))
            
        
    if x_to_x_correct:
        nx.draw_networkx_edges(G_disc, pos_disc, edgelist=x_to_x_correct,
                               edge_color='green', arrows=True,
                               arrowsize=20, width=2, ax=ax2, alpha=0.7)
    if x_to_x_incorrect:
        nx.draw_networkx_edges(G_disc, pos_disc, edgelist=x_to_x_incorrect,
                            edge_color='red', arrows=True,
                            arrowsize=20, width=2, ax=ax2, alpha=0.7,
                            style='dashed')
    if x_to_y_correct:
        nx.draw_networkx_edges(G_disc, pos_disc, edgelist=x_to_y_correct,
                            edge_color='green', arrows=True,
                            arrowsize=25, width=3, ax=ax2, alpha=0.8)
    if x_to_y_incorrect:
        nx.draw_networkx_edges(G_disc, pos_disc, edgelist=x_to_y_incorrect,
                            edge_color='red', arrows=True,
                            arrowsize=25, width=3, ax=ax2, alpha=0.8,
                            style='dashed')
    if u_to_any_edges:
        nx.draw_networkx_edges(G_disc, pos_disc, edgelist=u_to_any_edges,
                            edge_color='purple', arrows=True,
                            arrowsize=20, width=2, ax=ax2, alpha=0.7,
                            style='dashed')
        
    nx.draw_networkx_labels(G_disc, pos_disc, font_size=10, ax=ax2)

    ax2.set_title(f'{title_preix}\nDiscovered Caulsal Graph\n'
                  f'(Green= Correct, Red=Incorrect)',
                  fontsize=14)
    ax2.axis('off')

    plt.tight_layout()

    return fig
    
        
    