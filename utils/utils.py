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
    
        
    