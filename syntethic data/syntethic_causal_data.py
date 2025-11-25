import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt


class SyntheticCausalSystem:

    def __init__(self, n_features: int =10, random_state: Optional[int] = None, min_num_connected_edges: Optional[int] = 2, edge_probability: Optional[float]=0.3):

        self.n_features= n_features
        self.random_state = random_state
        self.min_num_connected_edges = min_num_connected_edges
        self.edge_probability=edge_probability
        if random_state is not None:
            np.random.seed(random_state)

        self.adjacency_matrix = None
        self.confounders_pais = []

    def _create_dag_structure(self) -> np.ndarray:

        adjacency = np.zeros((self.n_features,self.n_features))

        for i in range(self.n_features):
            for j in range(i +1, self.n_features):
                if np.random.random() < self.edge_probability:
                    adjacency[i,j] = 1
        
        # Make sure at least some edges exist

        if adjacency.sum() == 0: 
            # Add a few randome edges
            for _ in range(min(self.min_num_connected_edges, self.n_features-1)):
                i = np.random.randint(0,self.n_features - 1)
                j = np.random.randint(i+1,self.n_features)
                adjacency[i,j] = 1

        return adjacency
    
    def _add_confounders(self, n_confounders: int =2)->List[Tuple[int,List[int]]]:

        confounder_info=[]
        available_nodes= list(range(self.n_features))

        for _ in range(n_confounders):
            if len(available_nodes)<2:
                break

            n_affected = min(np.random.randint(2,4), len(available_nodes))
            affected = np.random.choice(available_nodes, size=n_affected, replace=False)

            confounder_info.append((len(confounder_info),affected.tolist()))

        return confounder_info
    
    def generate_linear_system(self,
                               n_samples: int = 1000,
                               with_confounders: bool= False,
                               noise_std: float = 0.5,
                               y_parents_ratio: float =0.4) -> Tuple[ pd.DataFrame, np.ndarray]:
        
        # Create Dag structure
        self.adjacency_matrix = self._create_dag_structure()

        # generate weight matrix 
        weights = self.adjacency_matrix * np.random.uniform(0.5,2.0,
                                                            (self.n_features, self.n_features))
        weights *= np.random.choice([-1,1], (self.n_features,self.n_features))

        # initialize data
        data = np.zeros((n_samples, self.n_features))

        # Add confounders if requested
        confounder_info = []

        if with_confounders:
            confounder_info = self._add_confounders(n_confounders=2)
            
            for conf_id, affected_nodes in confounder_info:
                confounder = np.random.rand(n_samples)

                for node in affected_nodes:

                    coef = np.random.uniform(0.5, 1.5) * np.random.choice([-1,1])
                    data[:, node] += coef * confounder
        
        # Generate data followin the causal structure

        for j in range(self.n_features):
            # Add contributions from paren nodes
            parents = np.where(weights[:,j]!=0)[0]
            for parent in parents:
                data[:,j] += weights[parent,j] * data[:, parent]

            # Add Gaussian noise
            data[:,j] += np.random.normal(0,noise_std,n_samples)


        # Generate outcome variable Y

        y = np.zeros(n_samples)

        # Select subset of features to affect Y

        n_parents = max(1, int(self.n_features * y_parents_ratio))
        y_parents_indices = np.random.choice(self.n_features, size=n_parents, replace=False)

        for parent_idx in y_parents_indices:
            coef = np.random.uniform(0.5,2.0) * np.random.choice([-1,1])
            y += coef * data[:, parent_idx]

        # add noise to Y
        y += np.random.normal(0, noise_std, n_samples)

        # Store Y parent indices for visualization
        self.y_parent_indices = y_parents_indices.tolist()

        columns = [f'X{i}' for i in range(self.n_features)] + ["Y"]
        df = pd.DataFrame(np.column_stack([data,y]),columns=columns)


        return df , self.adjacency_matrix, confounder_info
    

    def visualize_causal_graph(self, adjacency_matrix: np.ndarray,
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