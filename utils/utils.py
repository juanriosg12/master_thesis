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
                               alpha=0.9, ax=ax1, node_shape='s')
    
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


def plot_shapley_feature_comparison(shapley_dict, feature_names, 
                                    true_parents=None, title=None, figsize=(12, 8)):
    """
    Compare feature importance rankings across different Shapley methods.
    Shows mean absolute Shapley values for each feature across all instances.
    
    Parameters:
    -----------
    shapley_dict : dict
        Dictionary with method names as keys and Shapley values arrays as values
        Example: {'Library': shap_values_library, 'True': true_shapley_values, ...}
    feature_names : list
        List of feature names
    true_parents : list, optional
        List of true parent feature indices (will be marked with *)
    title : str, optional
        Custom title for the plot
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean absolute Shapley values per feature for each method
    n_features = len(feature_names)
    n_methods = len(shapley_dict)
    
    # Prepare data
    feature_importances = {method: np.mean(np.abs(vals), axis=0) 
                          for method, vals in shapley_dict.items()}
    
    # Set up bar positions
    x = np.arange(n_features)
    width = 0.8 / n_methods
    
    # Plot bars for each method
    colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
    
    for idx, (method_name, importances) in enumerate(feature_importances.items()):
        offset = (idx - n_methods/2 + 0.5) * width
        ax.bar(x + offset, importances, width, label=method_name, 
               color=colors[idx], alpha=0.8, edgecolor='black')
    
    # Customize plot
    feature_labels = feature_names.copy()
    if true_parents is not None:
        feature_labels = [f"{name}*" if i in true_parents else name 
                         for i, name in enumerate(feature_names)]
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean |Shapley Value|', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if title is None:
        title = 'Feature Importance Comparison Across Methods'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if true_parents is not None:
        ax.text(0.98, 0.98, '* = True Parent Feature', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


def plot_shapley_dependence(X, shapley_values, feature_names, method_name='SHAP',
                            true_parents=None, features_to_plot=None, 
                            figsize=(15, 10), ncols=3):
    """
    Create dependence plots showing SHAP values vs actual feature values.
    
    For each feature, creates a scatter plot where:
    - X-axis: actual feature values across all instances
    - Y-axis: corresponding SHAP values for that feature
    
    This helps understand how a feature's contribution changes with its value.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature values for all instances (n_instances, n_features)
    shapley_values : np.ndarray
        SHAP values (n_instances, n_features)
    feature_names : list
        List of feature names
    method_name : str, optional
        Name of the method for the title
    true_parents : list, optional
        List of true parent feature indices (will be highlighted with red borders)
    features_to_plot : list, optional
        Specific feature indices to plot. If None, plots all features
    figsize : tuple
        Figure size (width, height)
    ncols : int
        Number of columns in the subplot grid
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Convert X to numpy if needed
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    # Determine which features to plot
    n_features = X_values.shape[1]
    if features_to_plot is None:
        features_to_plot = range(n_features)
    
    n_plots = len(features_to_plot)
    nrows = int(np.ceil(n_plots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for plot_idx, feature_idx in enumerate(features_to_plot):
        ax = axes[plot_idx]
        
        # Get feature values and SHAP values
        x_vals = X_values[:, feature_idx]
        y_vals = shapley_values[:, feature_idx]
        
        # Create scatter plot with color based on SHAP value sign
        colors = np.where(y_vals >= 0, '#FF6B6B', '#4ECDC4')
        ax.scatter(x_vals, y_vals, c=colors, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Labels and title
        feature_name = feature_names[feature_idx]
        is_true_parent = true_parents is not None and feature_idx in true_parents
        
        if is_true_parent:
            title_str = f'{feature_name}*\n(True Parent)'
            ax.set_title(title_str, fontsize=10, fontweight='bold', color='darkred')
            # Add red border
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
        else:
            ax.set_title(feature_name, fontsize=10)
        
        ax.set_xlabel(f'{feature_name} value', fontsize=9)
        ax.set_ylabel(f'{method_name} value', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Remove unused subplots
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])
    
    # Main title
    fig.suptitle(f'{method_name} Dependence Plots: Feature Values vs SHAP Values', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    return fig


def plot_shapley_dependence_comparison(X, shapley_dict, feature_names, 
                                       feature_idx, true_parents=None,
                                       figsize=(15, 4)):
    """
    Compare SHAP value dependence on feature value across multiple methods.
    
    Creates side-by-side dependence plots for a single feature across all methods.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature values for all instances (n_instances, n_features)
    shapley_dict : dict
        Dictionary with method names as keys and Shapley values as values
    feature_names : list
        List of feature names
    feature_idx : int
        Index of the feature to plot
    true_parents : list, optional
        List of true parent feature indices
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    # Convert X to numpy if needed
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    n_methods = len(shapley_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)
    
    if n_methods == 1:
        axes = [axes]
    
    feature_name = feature_names[feature_idx]
    is_true_parent = true_parents is not None and feature_idx in true_parents
    
    x_vals = X_values[:, feature_idx]
    
    for idx, (method_name, shap_vals) in enumerate(shapley_dict.items()):
        ax = axes[idx]
        
        # Get SHAP values for this feature
        y_vals = shap_vals[:, feature_idx]
        
        # Create scatter plot with color based on SHAP value sign
        colors = np.where(y_vals >= 0, '#FF6B6B', '#4ECDC4')
        ax.scatter(x_vals, y_vals, c=colors, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Labels and title
        if is_true_parent:
            title_str = f'{method_name}\n{feature_name}* (True Parent)'
            ax.set_title(title_str, fontsize=10, fontweight='bold', color='darkred')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
        else:
            ax.set_title(f'{method_name}\n{feature_name}', fontsize=10)
        
        ax.set_xlabel(f'{feature_name} value', fontsize=9)
        if idx == 0:
            ax.set_ylabel('SHAP value', fontsize=9)
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Calculate and display correlation
        corr = np.corrcoef(x_vals, y_vals)[0, 1]
        ax.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=8)
    
    plt.tight_layout()
    
    return fig


def visualize_causal_graph_filtered_to_sink(adjacency_matrix: np.ndarray,
                                           feature_names: List[str],
                                           sink_name: str = 'Y',
                                           title: str = "Causal Graph (Filtered to Sink-Connected Nodes)",
                                           figsize=(14, 10),
                                           highlight_sources: bool = True,
                                           show_node_layers: bool = True):
    """
    Visualize only the nodes and edges that are connected to the sink node (Y).
    
    Uses backward BFS from the sink to identify all nodes that can reach it,
    then displays only the filtered subgraph.
    
    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        Adjacency matrix (n_features x n_features) including sink node Y
        Entry [i,j]=1 means feature i causes feature j
    feature_names : List[str]
        Names of all features INCLUDING the sink node (e.g., ['X0', 'X1', ..., 'Y'])
        The sink node should be the last element
    sink_name : str, default='Y'
        Name of the sink/target node to filter connections to
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    highlight_sources : bool, default=True
        If True, highlight source nodes (no incoming edges) in a different color
    show_node_layers : bool, default=True
        If True, use hierarchical layout showing distance from sources
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    G_filtered : nx.DiGraph
        The filtered NetworkX graph (for further analysis if needed)
    reachable_nodes : set
        Set of node names that are connected to sink
    real_source_indices : list
        Indices of real source nodes (can reach sink) for further analysis
    
    Examples:
    --------
    >>> # Create adjacency matrix (4x4 including Y at index 3)
    >>> adj = np.array([[0,1,0,0], [0,0,1,1], [0,0,0,1], [0,0,0,0]])
    >>> features = ['X0', 'X1', 'X2', 'Y']
    >>> fig, G, nodes = visualize_causal_graph_filtered_to_sink(adj, features)
    >>> plt.show()
    """
    
    # Validate inputs
    n_features = adjacency_matrix.shape[0]
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names length ({len(feature_names)}) must match "
                        f"adjacency_matrix size ({n_features})")
    
    if sink_name not in feature_names:
        raise ValueError(f"sink_name '{sink_name}' not found in feature_names")
    
    sink_idx = feature_names.index(sink_name)
    
    # Build parent dictionary from adjacency matrix
    parents = {}
    for j in range(n_features):
        parents[j] = [i for i in range(n_features) if adjacency_matrix[i, j] != 0]
    
    # Find all potential source nodes (no parents) - BEFORE filtering
    potential_sources = []
    for i in range(n_features):
        if len(parents[i]) == 0:
            potential_sources.append(i)
    
    # BACKWARD BFS FROM SINK - find all nodes that can reach the sink
    # This uses the SAME logic as ShapleyFlowWrapper for consistency
    reachable_indices = set()
    queue = [sink_idx]
    visited = {sink_idx}
    
    while queue:
        current_idx = queue.pop(0)
        reachable_indices.add(current_idx)
        
        # Add all parents (predecessors) of current node
        for parent_idx in parents.get(current_idx, []):
            if parent_idx not in visited:
                visited.add(parent_idx)
                queue.append(parent_idx)
    
    # Filter sources to only those that can reach the sink (REAL source nodes)
    # These are the actual starting points for causal paths to Y
    real_source_nodes = [s for s in potential_sources if s in reachable_indices]
    
    # Convert indices to feature names
    reachable_nodes = {feature_names[idx] for idx in reachable_indices}
    
    # Count filtered vs total
    n_total_nodes = n_features
    n_filtered_nodes = len(reachable_nodes)
    n_total_edges = int(np.sum(adjacency_matrix != 0))
    
    # Build NetworkX graph with ONLY reachable nodes and their edges
    G_filtered = nx.DiGraph()
    G_filtered.add_nodes_from(reachable_nodes)
    
    # Add edges between reachable nodes
    n_filtered_edges = 0
    for i in reachable_indices:
        for j in reachable_indices:
            if adjacency_matrix[i, j] != 0:
                G_filtered.add_edge(feature_names[i], feature_names[j])
                n_filtered_edges += 1
    
    # Identify node types for coloring
    source_nodes = []
    intermediate_nodes = []
    sink_nodes = []
    
    for node in reachable_nodes:
        node_idx = feature_names.index(node)
        
        if node == sink_name:
            sink_nodes.append(node)
        elif node_idx in real_source_nodes:
            # Real source nodes: no parents AND can reach sink
            source_nodes.append(node)
        else:
            # Intermediate nodes: have parents and/or children
            intermediate_nodes.append(node)
    
    # Create layout
    if show_node_layers and len(G_filtered.nodes()) > 0:
        # Try hierarchical layout (works best for DAGs)
        try:
            # Compute longest path from each node to sink for layering
            layers = {}
            for node in G_filtered.nodes():
                if node == sink_name:
                    layers[node] = 0
                else:
                    # BFS to find shortest path to sink
                    try:
                        path_length = nx.shortest_path_length(G_filtered, node, sink_name)
                        layers[node] = -path_length  # Negative for top-to-bottom layout
                    except nx.NetworkXNoPath:
                        layers[node] = -999  # Shouldn't happen with our filtering
            
            # Create hierarchical layout with more spacing
            # Increased k parameter for more distance between nodes
            pos = nx.spring_layout(G_filtered, k=3.0, iterations=100, seed=42)
            
            # Adjust y-coordinates based on layers
            for node, (x, y) in pos.items():
                layer = layers.get(node, 0)
                pos[node] = (x, layer)
                
        except:
            # Fallback to spring layout with more spacing
            pos = nx.spring_layout(G_filtered, k=3.0, iterations=100, seed=42)
    else:
        # More spread out spring layout
        pos = nx.spring_layout(G_filtered, k=3.0, iterations=100, seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw nodes by type
    if source_nodes:
        nx.draw_networkx_nodes(G_filtered, pos, nodelist=source_nodes,
                              node_color='lightgreen', node_size=1200,
                              alpha=0.9, ax=ax, label='Source Nodes')
    
    if intermediate_nodes:
        nx.draw_networkx_nodes(G_filtered, pos, nodelist=intermediate_nodes,
                              node_color='lightblue', node_size=1000,
                              alpha=0.9, ax=ax, label='Intermediate Nodes')
    
    if sink_nodes:
        nx.draw_networkx_nodes(G_filtered, pos, nodelist=sink_nodes,
                              node_color='gold', node_size=1500,
                              alpha=0.9, ax=ax, label='Sink Node (Y)')
    
    # Draw edges
    # Separate edges by type for better visualization
    edges_to_sink = [(u, v) for u, v in G_filtered.edges() if v == sink_name]
    edges_other = [(u, v) for u, v in G_filtered.edges() if v != sink_name]
    
    if edges_other:
        nx.draw_networkx_edges(G_filtered, pos, edgelist=edges_other,
                              edge_color='gray', arrows=True,
                              arrowsize=20, arrowstyle='->', width=2,
                              ax=ax, alpha=0.6)
    
    if edges_to_sink:
        nx.draw_networkx_edges(G_filtered, pos, edgelist=edges_to_sink,
                              edge_color='green', arrows=True,
                              arrowsize=25, arrowstyle='->', width=3,
                              ax=ax, alpha=0.8, label='Edges to Sink')
    
    # Draw labels
    nx.draw_networkx_labels(G_filtered, pos, font_size=11, 
                           font_weight='bold', ax=ax)
    
    # Add statistics text box
    stats_text = (f"Filtered Graph Statistics:\n"
                 f"Nodes: {n_filtered_nodes}/{n_total_nodes} "
                 f"({100*n_filtered_nodes/n_total_nodes:.1f}%)\n"
                 f"Edges: {n_filtered_edges}/{n_total_edges} "
                 f"({100*n_filtered_edges/max(1,n_total_edges):.1f}%)\n"
                 f"Real Sources: {len(real_source_nodes)}/{len(potential_sources)}")
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes, fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Title and legend
    ax.set_title(f"{title}\n(Only nodes connected to '{sink_name}')",
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Causal Graph Filtering Summary")
    print(f"{'='*60}")
    print(f"Sink node: {sink_name}")
    print(f"Total nodes in graph: {n_total_nodes}")
    print(f"Nodes connected to sink: {n_filtered_nodes} ({100*n_filtered_nodes/n_total_nodes:.1f}%)")
    print(f"Total edges in graph: {n_total_edges}")
    print(f"Edges in filtered graph: {n_filtered_edges} ({100*n_filtered_edges/max(1,n_total_edges):.1f}%)")
    print(f"\nPotential source nodes (no incoming edges): {len(potential_sources)}")
    if potential_sources:
        potential_source_names = [feature_names[i] for i in potential_sources]
        print(f"  {', '.join(sorted(potential_source_names))}")
    print(f"\nReal source nodes (can reach sink): {len(real_source_nodes)}")
    if real_source_nodes:
        real_source_names = [feature_names[i] for i in real_source_nodes]
        print(f"  {', '.join(sorted(real_source_names))}")
    
    # Show filtered-out sources (disconnected from sink)
    filtered_out_sources = [s for s in potential_sources if s not in real_source_nodes]
    if filtered_out_sources:
        print(f"\nFiltered source nodes (disconnected from sink): {len(filtered_out_sources)}")
        filtered_source_names = [feature_names[i] for i in filtered_out_sources]
        print(f"  {', '.join(sorted(filtered_source_names))}")
    
    print(f"\nIntermediate nodes: {len(intermediate_nodes)}")
    if intermediate_nodes:
        print(f"  {', '.join(sorted(intermediate_nodes))}")
    print(f"\nNodes filtered out (not connected to {sink_name}): {n_total_nodes - n_filtered_nodes}")
    if n_total_nodes - n_filtered_nodes > 0:
        filtered_out = [feature_names[i] for i in range(n_features) if i not in reachable_indices]
        print(f"  {', '.join(sorted(filtered_out))}")
    print(f"{'='*60}\n")
    
    return fig, G_filtered, reachable_nodes, real_source_nodes
        
    