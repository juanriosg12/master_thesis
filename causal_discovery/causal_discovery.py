import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, List, Dict
import warnings

# Import causal discovery algorithms from causal-learn
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import fisherz, kci, chisq
from causallearn.search.FCMBased.lingam import DirectLiNGAM
from causallearn.graph.Endpoint import Endpoint
from causallearn.search.FCMBased import lingam
from causallearn.utils.GraphUtils import GraphUtils

class CausalDiscoveryMethod:


    def __init__(self, alpha: float = 0.05, indep_test: str = 'fisherz'):

        self.alpha = alpha # significance level for independence test
        self.indep_test = indep_test # independece test to use ("fisherz", "kci", "chisq")
        self.discovered_graph = None
        self.confounders = []

    def _get_independence_test(self):
        if self.indep_test == 'fisherz':
            return fisherz
        elif self.indep_test == 'kci':
            return kci
        elif self.indep_test == 'chisq':
            return chisq
        else: 
            return fisherz
    
    def _has_cycle(self, adjacency: np.ndarray) -> bool:
        """
        Detect if a directed graph has cycles using Kahn's algorithm.
        
        Args:
            adjacency: Adjacency matrix where [i,j]=1 means edge i->j
            
        Returns:
            True if graph contains at least one cycle, False otherwise
        """
        n = adjacency.shape[0]
        in_degree = np.sum(adjacency != 0, axis=0).astype(int)
        queue = [i for i in range(n) if in_degree[i] == 0]
        visited = 0
        
        while queue:
            node = queue.pop(0)
            visited += 1
            for child in range(n):
                if adjacency[node, child] != 0:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
        
        return visited != n
    
    def _break_cycles_dfs(self, adjacency: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """
        Remove edges to break cycles using depth-first search.
        
        Args:
            adjacency: Adjacency matrix where [i,j]=1 means edge i->j
            feature_names: Optional list of feature names for logging
            
        Returns:
            Adjacency matrix with cycles broken (acyclic)
        """
        n = adjacency.shape[0]
        adj_copy = adjacency.copy()
        visited = [0] * n  # 0: unvisited, 1: visiting, 2: visited
        removed_edges = []
        
        def dfs(node):
            visited[node] = 1  # Mark as visiting
            for child in range(n):
                if adj_copy[node, child] != 0:
                    if visited[child] == 1:  # Back edge detected - cycle!
                        adj_copy[node, child] = 0  # Remove edge
                        if feature_names:
                            removed_edges.append((feature_names[node], feature_names[child]))
                        else:
                            removed_edges.append((node, child))
                    elif visited[child] == 0:
                        dfs(child)
            visited[node] = 2  # Mark as visited
        
        for i in range(n):
            if visited[i] == 0:
                dfs(i)
        
        # Log removed edges
        if removed_edges:
            print(f"   Removed {len(removed_edges)} edge(s) to break cycles:")
            for source, target in removed_edges:
                print(f"      {source} -> {target}")
        
        return adj_copy
        
    def _extract_adjacency_from_graph(self, graph, n_features: int) -> np.ndarray:
        """
        Extract adjacency matrix from PC graph object.
        
        PC returns a CPDAG with special encoding:
        - graph[i,j]=-1 and graph[j,i]=1  → directed edge i->j
        - graph[i,j]=-1 and graph[j,i]=-1 → undirected edge i-j
        - graph[i,j]=1 and graph[j,i]=1   → bidirected edge i<->j (confounder)
        - graph[i,j]=0 and graph[j,i]=0   → no edge
        
        We convert this to a standard directed adjacency matrix.
        """
        adjacency = np.zeros((n_features, n_features))
        
        try:
            pc_graph = graph.graph
            if pc_graph.shape[0] != n_features:
                return adjacency
            
            # Process each potential edge
            for i in range(n_features):
                for j in range(i + 1, n_features):  # Only check upper triangle
                    edge_ij = pc_graph[i, j]
                    edge_ji = pc_graph[j, i]
                    
                    # Directed edge i -> j
                    if edge_ij == -1 and edge_ji == 1:
                        adjacency[i, j] = 1
                    
                    # Directed edge j -> i
                    elif edge_ij == 1 and edge_ji == -1:
                        adjacency[j, i] = 1
                    
                    # # Undirected edge i - j (orient arbitrarily as i -> j)
                    # elif edge_ij == -1 and edge_ji == -1:
                    #     adjacency[i, j] = 1
                    
                    # # Bidirected edge i <-> j (confounder - only add one direction to avoid cycle)
                    # # The confounders will be tracked separately via FCI
                    # elif edge_ij == 1 and edge_ji == 1:
                    #     adjacency[i, j] = 1
                    #     # Do NOT add adjacency[j, i] = 1 to avoid creating a cycle
                        
        except Exception as e:
            warnings.warn(f"Error extracting adjacency from graph: {str(e)}")
            adjacency = np.zeros((n_features, n_features))
        
        # Check for and break cycles
        if self._has_cycle(adjacency):
            warnings.warn("Graph contains cycles. Breaking cycles to ensure DAG structure.")
            adjacency = self._break_cycles_dfs(adjacency, feature_names=None)
        
        return adjacency
    
    def _detect_confounders_from_graph(self, graph, feature_names: List[str]) -> List[Tuple[str, str]]:
        # Detect the confounders from a PAG using FCI

        confounders = []
        n = len(feature_names)

        try:

            for i in range(n):
                for j in range(i + 1, n):
                    edge = graph.get_edge(graph.nodes[i], graph.nodes[j])

                    if edge is not None:
                        endpoint_i = edge.get_endpoint1()
                        endpoint_j = edge.get_endpoint2()

                        if endpoint_i == Endpoint.ARROW and endpoint_j == Endpoint.ARROW:
                            confounders.append((feature_names[i], feature_names[j]))
        except Exception as e:
            warnings.warn(f"Could not detect confounders: {str(e)}")

        return confounders
    
    def visualize_discovered_graph(self, adjacency: np.ndarray,
                                   feature_names: List[str],
                                   confounders: List = None,
                                   title: str = "Discovered Causal Graph"):
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(feature_names)

        # Add edges from adjacency matrix
        n = len(feature_names)
        for i in range(n):
            for j in range(n):
                if adjacency[i,j] != 0:
                    G.add_edge(feature_names[i],feature_names[j])

        # Create layout 
        pos = nx.spring_layout(G,seed=42)

        # Draw the graph
        plt.figure(figsize=(12,8))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen',
                               node_size=600, alpha=0.9)
        
        # Draw edgeds
        nx.draw_networkx_edges(G, pos, edgecolors='gray',
                               arrows=True, arrowsize=20,
                               arrowstyle='->',width=2)
        
        # Highlight confounder pairs
        if confounders:
            for pair in confounders:
                if len(pair) == 2:
                    # Draw bidirected edge for confounders
                    nx.draw_networkx_edges(G, pos,
                                           [(pair[0],pair[1])],
                                           edge_color="red",
                                           style="dashed",
                                           arrows=True,
                                           arrowsize=15,
                                           width=2)
        
        # Draw labels
        nx.draw_networkx_labels(G,pos, font_size=10)

        plt.title(title, fontsize=14)
        plt.axis("off")
        plt.tight_layout()

        return plt
    
class PCWithFCI(CausalDiscoveryMethod):


    def __init__(self,alpha: float = 0.05, indep_test: str= 'fisherz'):

        super().__init__(alpha=alpha, indep_test=indep_test)
        self.pc_result= None
        self.fci_result = None

    def discover_structure(self, data: pd.DataFrame) -> Tuple[np.ndarray, List]:
        
        print(" Running PC algorithm for causal structure discovery...")
        data_array = data.values
        feature_names = data.columns.tolist()
        n_features = len(feature_names)

        indep_test_func = self._get_independence_test()

        try:
            # PC algorithm for causal discovery
            self.pc_result = pc(
                data_array,
                alpha=self.alpha,
                indep_test=indep_test_func,
                stable=True,
                uc_rule=0,
                uc_priority=2
            )
            # Extract adjacency matrix from PC result
            pc_graph = self.pc_result.G
            adjacency_pc = self._extract_adjacency_from_graph(pc_graph, n_features)
            
            print(f"PC discovered {np.sum(adjacency_pc != 0)} edges (after cycle removal)")

        except Exception as e:
            print(f"PC algorithm failed: {str(e)}")
            adjacency_pc = np.zeros((n_features, n_features))

        # Run FCI for confounder detection
        print(" Running FCI algorithm for confounder detection...")

        try:
            self.fci_result, edges = fci(
                data_array,
                alpha=self.alpha,
                indep_test=indep_test_func,
                stable=True
            )

            # Extract confounders from FCI result
            fci_graph = self.fci_result
            confounders= self._detect_confounders_from_graph(fci_graph, feature_names)

            print(f"FCI detected {len(confounders)} potential confounders pairs")
        
        except Exception as e:
            print(f"FCI algorithm failed: {str(e)}")
            confounders = []

        self.discovered_graph = adjacency_pc
        self.confounders = confounders


        return adjacency_pc, confounders
        
    def get_causal_relationships(self, data: pd.DataFrame) -> Dict:

        adjacency, confounders = self.discover_structure(data)

        results = {
            'method': 'PC + FCI',
            'adjacency_matrix': adjacency,
            'confounders': confounders,
            'feature_names': data.columns.tolist(),
            'n_edges': np.sum(adjacency!=0),
            'n_confounders_pairs': len(confounders)
        }

        return results

class LiNGAMWithFCI(CausalDiscoveryMethod):

    def __init__(self,alpha: float = 0.05, indep_test: str= 'fisherz'):

        super().__init__(alpha=alpha, indep_test=indep_test)
        self.lingam_result= None
        self.fci_result = None
    
    def discover_structure(self, data: pd.DataFrame) -> Tuple[np.ndarray, List]:

        print("Running LiNGAM algorithm for causal structure discovery...")

        data_array = data.values
        feature_names = data.columns.tolist()
        n_features = len(feature_names)

        # Run LiNGAM algorithm
        try:
            model = DirectLiNGAM()
            model.fit(data_array)

            adjacency_lingam = model.adjacency_matrix_
            
            threashold = 0.1
            adjacency_lingam[np.abs(adjacency_lingam) < threashold] = 0

            adjacency_binary = (np.abs(adjacency_lingam) > 0).astype(int)

            print(f"LiNGAM discovered {np.sum(adjacency_binary != 0)} edges")

            self.lingam_result = model
        except Exception as e:
            print(f"LiNGAM algorithm failed: {str(e)}")
            print(f"Trying ICA-based LiNGAm as fallback...")
            try:
                # Fallback to ICA-based LiNGAM
                from causallearn.search.FCMBased.lingam import ICALiNGAM
                
                model = ICALiNGAM()
                model.fit(data_array)
                adjacency_lingam = model.adjacency_matrix_
                
                threashold = 0.01
                adjacency_lingam[np.abs(adjacency_lingam) < threashold] = 0
                adjacency_binary = (np.abs(adjacency_lingam) > 0).astype(int)

                print(f"ICA-LiNGAM discovered {np.sum(adjacency_binary !=0 )} edges")
                self.lingam_result = model

            except Exception as e2:
                print(f"ICA-LiNGAM also failed: {str(e2)}")
                adjacency_binary = np.zeros((n_features, n_features))
        
        # Run FCI algorithm for confounder detection
        print("Running FCI algorithm for confounder detection...")


        indep_test_func = self._get_independence_test()

        try:
            self.fci_result, edges = fci(
                data_array,
                alpha=self.alpha,
                indep_test=indep_test_func,
                stable=True
            )

            # Extract confounders from FCI result
            fci_graph = self.fci_result
            confounders= self._detect_confounders_from_graph(fci_graph, feature_names)

            print(f"FCI detected {len(confounders)} potential confounders pairs")
        
        except Exception as e:
            print(f"FCI algorithm failed: {str(e)}")
            confounders = []

        self.discovered_graph = adjacency_binary
        self.confounders = confounders

        return adjacency_binary, confounders
    
    def get_causal_relationships(self, data: pd.DataFrame) -> Dict:


        adjacency, confounders = self.discover_structure(data)

        results = {
            'method': 'LiNGAM + FCI',
            'adjacency_matrix': adjacency,
            'confounders': confounders,
            'feature_names': data.columns.tolist(),
            'n_edges': np.sum(adjacency!=0),
            'n_confounders_pairs': len(confounders)
        }

        return results 


def compare_with_ground_truth( discovered_adj: np.ndarray,
                              true_adj: np.ndarray,
                              discovered_confounders: List,
                              true_confounders: List) -> Dict:
    # Edge detection metrics

    true_edges = (true_adj !=0).astype(int)
    pred_edges = (discovered_adj !=0).astype(int)

    # True positives, false positives, false negatives
    tp = np.sum((true_edges ==1) & (pred_edges ==1))
    fp = np.sum((true_edges ==0) & (pred_edges ==1))
    fn = np.sum((true_edges ==1) & (pred_edges ==0))
    tn = np.sum((true_edges ==0) & (pred_edges ==0))

    # Calculate metrics
    precision = tp /(tp +fp) if (tp +fp) > 0 else 0
    recall = tp /(tp +fn) if (tp +fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    shd = fp+ fn

    metrics = {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'structural_hamming_distance': int(shd),
        'n_true_edges': int(true_edges.sum()),
        'n_discovered_edges': int(pred_edges.sum()),
        'n_true_confounders': len(true_confounders),
        'n_discovered_confounders': len(discovered_confounders)
    } 

    return metrics