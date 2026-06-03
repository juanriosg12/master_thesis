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
        self.y_generation_params = None  # Will store Y generation parameters
    
    def _normalize_variable(self, values: np.ndarray, clip_std: float = 5.0) -> np.ndarray:
        """Normalize a variable by clipping outliers and standardizing.
        
        Args:
            values: Array of values to normalize
            clip_std: Number of standard deviations for clipping outliers
            
        Returns:
            Normalized values with mean=0 and std=1
        """
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val > 1e-10:  # Avoid division by zero
            # Clip extreme outliers
            clipped = np.clip(values, mean_val - clip_std * std_val, mean_val + clip_std * std_val)
            # Standardize to unit variance
            normalized = (clipped - np.mean(clipped)) / np.std(clipped)
            return normalized
        else:
            return values - mean_val  # Just center if std is too small

    def _create_dag_structure(self) -> np.ndarray:
        """Generate a random DAG with uniformly distributed causal order.

        Step 1: Build an upper-triangular skeleton (Erdős–Rényi on n*(n-1)/2
        candidate edges) — this guarantees acyclicity for the canonical
        ordering 0 < 1 < … < n-1.

        Step 2: Apply a random permutation P to both rows and columns.  The
        resulting adjacency A' = P A P^T represents the same DAG under a new
        (random) variable labelling, so edges are no longer confined to the
        upper triangle of the index-ordered matrix.
        """
        n = self.n_features

        # --- Step 1: upper-triangular skeleton ---
        skeleton = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.random() < self.edge_probability:
                    skeleton[i, j] = 1

        # Fallback: guarantee at least min_num_connected_edges
        if skeleton.sum() == 0:
            for _ in range(min(self.min_num_connected_edges, n - 1)):
                i = np.random.randint(0, n - 1)
                j = np.random.randint(i + 1, n)
                skeleton[i, j] = 1

        # --- Step 2: apply random permutation to node labels ---
        perm = np.random.permutation(n)           # e.g. [3, 0, 7, 1, …]
        adjacency = skeleton[np.ix_(perm, perm)]  # permute rows AND columns

        # Store the topological order so generate_* methods can use it.
        # perm[k] = original node index of the k-th node in causal order, so
        # iterating nodes in the order given by np.argsort(perm) traverses
        # them from sources to sinks in the permuted graph.
        self._topo_order = np.argsort(perm).tolist()  # topo order in permuted indices

        return adjacency

    def _topological_order(self, adjacency: np.ndarray) -> list:
        """Return node indices in topological order using Kahn's algorithm.

        Works for any DAG; serves as a fallback when _topo_order is not
        available (e.g. if adjacency was supplied externally).
        """
        n = adjacency.shape[0]
        in_deg = adjacency.sum(axis=0).astype(int)  # column sum = in-degree
        queue = [i for i in range(n) if in_deg[i] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in range(n):
                if adjacency[node, child]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
        if len(order) != n:  # cycle guard (should never happen for a DAG)
            return list(range(n))
        return order
    
    def _add_confounders(self, n_confounders: int =5)->List[Tuple[int,List[int]]]:

        confounder_info=[]
        available_nodes= list(range(self.n_features))

        n_confounders = min(n_confounders, int(self.n_features * 0.2))  # Limit number of confounders to avoid excessive overlap

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
            confounder_info = self._add_confounders()
            
            for conf_id, affected_nodes in confounder_info:
                confounder = np.random.rand(n_samples)

                for node in affected_nodes:

                    coef = np.random.uniform(0.5, 1.5) * np.random.choice([-1,1])
                    data[:, node] += coef * confounder
        
        # Generate data following the causal structure

        for j in range(self.n_features):
            # Add contributions from parent nodes
            parents = np.where(weights[:,j]!=0)[0]
            for parent in parents:
                data[:,j] += weights[parent,j] * data[:, parent]
            
            # Normalize after accumulating parent contributions to prevent explosion
            if len(parents) > 0:
                data[:,j] = self._normalize_variable(data[:,j])

            # Add Gaussian noise
            data[:,j] += np.random.normal(0,noise_std,n_samples)


        # Generate outcome variable Y

        y = np.zeros(n_samples)

        # Select subset of features to affect Y

        n_parents = max(1, int(self.n_features * y_parents_ratio))
        y_parents_indices = np.random.choice(self.n_features, size=n_parents, replace=False)

        # Store Y generation parameters (coefficients for each parent)
        y_coefficients = {}
        for parent_idx in y_parents_indices:
            coef = np.random.uniform(0.5,2.0) * np.random.choice([-1,1])
            y_coefficients[int(parent_idx)] = coef
            y += coef * data[:, parent_idx]

        # add noise to Y
        y += np.random.normal(0, noise_std, n_samples)

        # Store Y parent indices for visualization
        self.y_parent_indices = y_parents_indices.tolist()
        
        # Store Y generation parameters for ground truth Shapley computation
        self.y_generation_params = {
            'type': 'linear',
            'coefficients': y_coefficients,
            'parent_indices': y_parents_indices.tolist(),
            'noise_std': noise_std,
            'weights': weights  # Store the full weight matrix for feature dependencies
        }

        # Extend adjacency matrix to include Y variable
        complete_adjacency = np.zeros((self.n_features + 1, self.n_features + 1))
        complete_adjacency[:self.n_features, :self.n_features] = self.adjacency_matrix

        # Add edges from X variables to Y
        for parent_idx in y_parents_indices:
            complete_adjacency[parent_idx, self.n_features] = 1

        columns = [f'X{i}' for i in range(self.n_features)] + ["Y"]
        df = pd.DataFrame(np.column_stack([data,y]),columns=columns)


        return df , complete_adjacency, confounder_info
    
    def generate_nonlinear_system(self,
                               n_samples: int = 1000,
                               with_confounders: bool= False,
                               noise_std: float = 0.5,
                               y_parents_ratio: float =0.4) -> Tuple[ pd.DataFrame, np.ndarray]:
        
        # Create Dag structure
        self.adjacency_matrix = self._create_dag_structure()

        # initialize data
        data = np.zeros((n_samples, self.n_features))

        # Add confounders if requested
        confounder_info = []
        if with_confounders:
            confounder_info = self._add_confounders()

            for conf_id, affected_nodes in confounder_info:
                # Generate confounder variable
                confounder = np.random.randn(n_samples)

                # Affect the specified nodes with non-linear relationships
                for node in affected_nodes:
                    func_type = np.random.choice(["square","tanh","exp"])
                    coef = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])

                    if func_type == 'square':
                        data[:, node] += coef * confounder ** 2
                    elif func_type == 'tanh':
                        data[:, node] += coef * np.tanh(confounder)
                    else:
                        data[:, node] += coef * (np.exp(confounder / 2) - 1)

        # Generate data following the causal structure with non-linear functions
        # Iterate in topological order so parents are always computed first
        topo = getattr(self, '_topo_order', None) or self._topological_order(self.adjacency_matrix)

        for j in topo:
            parents = np.where(self.adjacency_matrix[:,j] == 1)[0]

            for parent in parents:
                # Use different non-linear functions
                func_type = np.random.choice(["square","cube","tanh","sin"])
                coef = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])

                if func_type == 'square':
                    data[:, j] += coef * data[:, parent] ** 2
                elif func_type == 'cube':
                    data[:, j] += coef * data[:, parent] ** 3
                elif func_type == 'tanh':
                    data[:, j] += coef * np.tanh(data[:, parent])
                else:
                    data[:, j] += coef * np.sin(data[:, parent])
            
            # Normalize after accumulating parent contributions to prevent explosion
            if len(parents) > 0:
                data[:, j] = self._normalize_variable(data[:, j])

            # Add Gaussian noise 
            data[:, j] += np.random.normal(0, noise_std, n_samples)

        # Generate outcome variable Y with non-linear relationships
        y = np.zeros(n_samples)

        # Select subset of features to affect Y
        n_parents= max(1, int(self.n_features * y_parents_ratio))
        y_parents_indices = np.random.choice(self.n_features, size=n_parents, replace=False)

        # Y is affected by selected parents with non-linear relationships
        y_coefficients = {}
        y_func_types = {}
        for parent_idx in y_parents_indices:
            func_type = np.random.choice(["square","cube","tanh","sin"])
            # Use smaller coefficients for extreme functions
            if func_type in ["square", "cube"]:
                coef = np.random.uniform(0.1, 0.3) * np.random.choice([-1, 1])
            else:
                coef = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
            y_coefficients[int(parent_idx)] = coef
            y_func_types[int(parent_idx)] = func_type

            if func_type == 'square':
                y += coef * data[:, parent_idx] ** 2
            elif func_type == 'cube':
                y += coef * data[:,parent_idx] ** 3
            elif func_type == 'tanh':
                y += coef * np.tanh(data[:, parent_idx])
            else:
                y += coef * np.sin(data[:, parent_idx])
        
        # Normalize Y to prevent extreme values
        if len(y_parents_indices) > 0:
            y = self._normalize_variable(y)
        
        # Add noise to Y
        y += np.random.normal(0 ,noise_std, n_samples)

        # Store Y parent indices for visualization
        self.y_parent_indices = y_parents_indices.tolist()
        
        # Store Y generation parameters for ground truth Shapley computation
        self.y_generation_params = {
            'type': 'nonlinear',
            'coefficients': y_coefficients,
            'func_types': y_func_types,
            'parent_indices': y_parents_indices.tolist(),
            'noise_std': noise_std
        }
        
        # Extend adjacency matrix to include Y variable
        complete_adjacency = np.zeros((self.n_features + 1, self.n_features + 1))
        complete_adjacency[:self.n_features, :self.n_features] = self.adjacency_matrix

        # Add edges from X variables to Y
        for parent_idx in y_parents_indices:
            complete_adjacency[parent_idx, self.n_features] = 1

        # Create DataFrame with Y
        columns = [f"X{i}" for i in range(self.n_features)] +["Y"]
        df = pd.DataFrame(np.column_stack([data, y]), columns=columns)

        return df, complete_adjacency, confounder_info
        
    def generate_mixed_system(self,
                               n_samples: int = 1000,
                               with_confounders: bool= False,
                               noise_std: float = 0.5,
                               linear_ratio: float = 0.5,
                               y_parents_ratio: float =0.4) -> Tuple[ pd.DataFrame, np.ndarray]:
        
        # Create Dag structure
        self.adjacency_matrix = self._create_dag_structure()

        # initialize data
        data = np.zeros((n_samples, self.n_features))

        # Determinmde wich edges are linear vs non-linear
        edges = np.argwhere(self.adjacency_matrix==1)
        n_edges = len(edges)
        n_linear = int(n_edges * linear_ratio)

        # Randomly assing edges to be linear or non-lienar
        edge_types = ["linear"] * n_linear + ["nonlinear"] * (n_edges -n_linear)
        np.random.shuffle(edge_types)

        # Add confounders if requested
        confounder_info = []
        if with_confounders:
            confounder_info = self._add_confounders()

            for  conf_id, affected_nodes in confounder_info:
                # Generate confounder variable
                confounder = np.random.randn(n_samples)

                for idx, node in enumerate(affected_nodes):
                    if  idx % 2 == 0:
                        coef = np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1])
                        data[:, node] += coef * confounder
                    else: # Non-linear
                        coef = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
                        data[:, node] += coef * np.tanh(confounder)
        # Generate data following the causal structure
        # Create a dictionary mapping (parent, child) -> (edge_type, func_type, coef)
        # This allows us to iterate by child (like linear/nonlinear) while preserving edge types
        edge_params = {}
        for edge_idx, (parent, child) in enumerate(edges):
            edge_type = edge_types[edge_idx]
            
            if edge_type == 'linear':
                coef = np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1])
                edge_params[(parent, child)] = ('linear', None, coef)
            else:  # non-linear
                func_type = np.random.choice(["square", "cube", "tanh", "sin"])
                coef = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
                edge_params[(parent, child)] = ('nonlinear', func_type, coef)
        
        # Iterate through features (children) in topological order so parents are computed first
        topo = getattr(self, '_topo_order', None) or self._topological_order(self.adjacency_matrix)

        for j in topo:
            # Find all parents of this child
            parents = np.where(self.adjacency_matrix[:, j] == 1)[0]
            
            # Accumulate contributions from all parents BEFORE normalizing
            for parent in parents:
                edge_type, func_type, coef = edge_params[(parent, j)]
                
                if edge_type == 'linear':
                    data[:, j] += coef * data[:, parent]
                else:  # non-linear
                    if func_type == 'square':
                        data[:, j] += coef * data[:, parent] ** 2
                    elif func_type == 'cube':
                        data[:, j] += coef * data[:, parent] ** 3
                    elif func_type == 'tanh':
                        data[:, j] += coef * np.tanh(data[:, parent])
                    else:  # sin
                        data[:, j] += coef * np.sin(data[:, parent])
            
            # Normalize after accumulating ALL parent contributions (like linear/nonlinear)
            if len(parents) > 0:
                data[:, j] = self._normalize_variable(data[:, j])
            
            # Add Gaussian noise
            data[:, j] += np.random.normal(0, noise_std, n_samples)

        # Generate outcome variable Y with mixed relationship
        y = np.zeros(n_samples)

        # Select subset of features to affect Y

        n_parents = max(1, int(self.n_features * y_parents_ratio))
        y_parents_indices = np.random.choice(self.n_features, size= n_parents, replace=False)

        # Y is affected by selected parents with mixed linear/non-linear relationships
        y_coefficients = {}
        y_func_types = {}
        for idx, parent_idx in enumerate(y_parents_indices):
            if idx % 2 ==0: # linear
                coef = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
                y_coefficients[int(parent_idx)] = coef
                y_func_types[int(parent_idx)] = 'linear'
                y += coef * data[:, parent_idx]
            else: # Non-linear
                func_type = np.random.choice(["square","cube","tanh","sin"])
                # Use smaller coefficients for extreme functions
                if func_type in ["square", "cube"]:
                    coef = np.random.uniform(0.1, 0.3) * np.random.choice([-1, 1])
                else:
                    coef = np.random.uniform(0.3, 0.8) * np.random.choice([-1, 1])
                y_coefficients[int(parent_idx)] = coef
                y_func_types[int(parent_idx)] = func_type

                if func_type == 'square':
                     y += coef * data[:, parent_idx] ** 2
                elif func_type == 'cube':
                    y += coef * data[:,parent_idx] ** 3
                elif func_type == 'tanh':
                    y += coef * np.tanh(data[:, parent_idx])
                else:
                    y += coef * np.sin(data[:, parent_idx])
        
        # Normalize Y to prevent extreme values
        if len(y_parents_indices) > 0:
            y = self._normalize_variable(y)

        # Add noise to Y
        y += np.random.normal(0, noise_std, n_samples)

        # store Y parent indices for visualization
        self.y_parent_indices = y_parents_indices.tolist()
        
        # Store Y generation parameters for ground truth Shapley computation
        self.y_generation_params = {
            'type': 'mixed',
            'parent_indices': y_parents_indices.tolist(),
            'coefficients': y_coefficients,
            'func_types': y_func_types,
            'noise_std': noise_std
        }

        # Extend adjacency matrix to include Y variable
        complete_adjacency = np.zeros((self.n_features + 1, self.n_features + 1))
        complete_adjacency[:self.n_features, :self.n_features] = self.adjacency_matrix

        # Add edges from X variables to Y
        for parent_idx in y_parents_indices:
            complete_adjacency[parent_idx, self.n_features] = 1

        # Create DataFrame with Y
        columns = [f"X{i}" for i in range(self.n_features)] +["Y"]
        df = pd.DataFrame(np.column_stack([data, y]), columns=columns)

        # TODO: save the relationships and coeficient of each x variable to Y, this is would be the real value of the shap_value 
        return df, complete_adjacency, confounder_info, edge_types

    def get_true_y_generator(self):
        """
        Returns a function that computes Y from X using the true generation parameters.
        This function can be used to compute ground truth Shapley values.
        
        Returns:
            callable: A function that takes an array of shape (n_samples, n_features) 
                     and returns Y values of shape (n_samples,)
        """
        if self.y_generation_params is None:
            raise ValueError("No Y generation parameters found. Generate data first.")
        
        params = self.y_generation_params
        parent_indices = params['parent_indices']
        coefficients = params['coefficients']
        gen_type = params['type']
        
        if gen_type == 'linear':
            def generator(X):
                """Linear generator: Y = sum(coef_i * X_i)"""
                y = np.zeros(X.shape[0])
                for parent_idx in parent_indices:
                    y += coefficients[parent_idx] * X[:, parent_idx]
                return y
                
        elif gen_type == 'nonlinear':
            func_types = params['func_types']
            
            def generator(X):
                """Nonlinear generator: Y = sum(coef_i * f_i(X_i))"""
                y = np.zeros(X.shape[0])
                for parent_idx in parent_indices:
                    coef = coefficients[parent_idx]
                    func_type = func_types[parent_idx]
                    if func_type == 'square':
                        y += coef * X[:, parent_idx] ** 2
                    elif func_type == 'cube':
                        y += coef * X[:, parent_idx] ** 3
                    elif func_type == 'tanh':
                        y += coef * np.tanh(X[:, parent_idx])
                    else:  # sin
                        y += coef * np.sin(X[:, parent_idx])
                return y
                
        else:  # mixed
            func_types = params['func_types']
            
            def generator(X):
                """Mixed generator: Y = sum(coef_i * f_i(X_i)) with linear and nonlinear f_i"""
                y = np.zeros(X.shape[0])
                for parent_idx in parent_indices:
                    coef = coefficients[parent_idx]
                    func_type = func_types[parent_idx]
                    if func_type == 'linear':
                        y += coef * X[:, parent_idx]
                    elif func_type == 'square':
                        y += coef * X[:, parent_idx] ** 2
                    elif func_type == 'cube':
                        y += coef * X[:, parent_idx] ** 3
                    elif func_type == 'tanh':
                        y += coef * np.tanh(X[:, parent_idx])
                    else:  # sin
                        y += coef * np.sin(X[:, parent_idx])
                return y
        
        return generator

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