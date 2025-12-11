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
        
    def _extract_adjacency_from_graph(self, graph, n_features: int) -> np.ndarray:

        try:
            adjacency = graph.graph
            if adjacency.shape[0] != n_features:
                adjacency = np.zeros((n_features, n_features))
        except:
            adjacency = np.zeros((n_features, n_features))
        
        return adjacency
    
    def _detect_confounders_from_graph(self, graph, feature_names: List[str]) -> List[Tuple[str, str]]:
        # Detect the confounders from a PAG using FCI

        confounders = []
        n = len(feature_names)

        try:
            # Get the graph matrix
            graph_matrix = graph.graph

            for i in range(n):
                for j in range(i + 1, n):
                    # Check for bidirected edge pattern
                    # -1 indicates no edge, 1 indicates edge with specific endpoint
                    if graph_matrix[i,j] !=0 and graph_matrix[j,i] !=0:
                        # Potential bidirected edge suggesting confounder
                        confounders.append((feature_names[i],feature_names[j]))
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


    def __init_(self,alpha: float = 0.05, indep_test: str= 'fisherz'):

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

            print(f"PC discovered {np.sum(adjacency_pc != 0)} edges")

        except Exception as e:
            print(f"PC algorithm failed: {str(e)}")
            adjacency_pc = np.zeros((n_features, n_features))

        # Run FCI for confounder detection
        print(" Running FCI algorithm for confounder detection...")

        try:
            self.fci_result = fci(
                data_array,
                alpha=self.alpha,
                independence_test_method=indep_test_func,
                stable=True
            )

            # Extract confounders from FCI result
            fci_graph = self.fci_result.G
            confounders= self._detect_confounders_from_graph(fci, feature_names)

            print(f"FCI detected {len(confounders)} potential confounders pairs")
        
        except Exception as e:
            print(f"FCI detected {len(confounders)} potential confounder pairs")
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
            'features_names': data.columns.tolist(),
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