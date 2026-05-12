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
                    
                    # Undirected edge i - j: PC cannot determine orientation.
                    # Orient arbitrarily as i -> j (lower index to higher index).
                    # This preserves the skeleton (recall) at the cost of ~50%
                    # direction errors on these edges — better than dropping them
                    # entirely which causes massive false negatives.
                    elif edge_ij == -1 and edge_ji == -1:
                        adjacency[i, j] = 1
                    
                    # Bidirected i <-> j: confounder indicator from PAG.
                    # Add one direction only (avoid creating a cycle).
                    # These are also captured in the confounders list via FCI.
                    elif edge_ij == 1 and edge_ji == 1:
                        adjacency[i, j] = 1
                        
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

    def _determine_y_parents(self, x_adjacency: np.ndarray,
                              x_feature_names: List[str],
                              model=None) -> List[int]:
        """Determine which X features should have edges to Y.

        Y edges are set for the union of:
        1. Features used by the ML model (model.selected_features) — "model world"
        2. Sink nodes in the X-only graph (no outgoing edges) — ensures Y is
           reachable from every causal path

        If model is None, only rule 2 applies.  If neither rule produces any
        parents (e.g. dense graph with no sinks and no model), all features are
        used as a safe fallback.
        """
        n_features = len(x_feature_names)
        y_parents: set = set()

        # Rule 1: features that the ML model actually uses
        if (model is not None
                and hasattr(model, 'selected_features')
                and model.selected_features is not None):
            feat_index = {name: idx for idx, name in enumerate(x_feature_names)}
            for feat_name in model.selected_features:
                if feat_name in feat_index:
                    y_parents.add(feat_index[feat_name])

        # Rule 2: sink nodes (no outgoing edges to other X features)
        for i in range(n_features):
            if x_adjacency[i, :].sum() == 0:
                y_parents.add(i)

        # Fallback: connect everything so the graph is not disconnected
        if len(y_parents) == 0:
            y_parents = set(range(n_features))

        return sorted(y_parents)

    def _build_full_adjacency(self, x_adjacency: np.ndarray,
                               x_feature_names: List[str],
                               model=None) -> np.ndarray:
        """Build an (n_x+1) × (n_x+1) adjacency matrix that includes Y.

        The X-to-X block comes from the discovery algorithm; X-to-Y edges are
        determined by :meth:`_determine_y_parents`.
        """
        n_features = len(x_feature_names)
        full_adj = np.zeros((n_features + 1, n_features + 1))
        full_adj[:n_features, :n_features] = x_adjacency

        for parent_idx in self._determine_y_parents(x_adjacency, x_feature_names, model):
            full_adj[parent_idx, n_features] = 1

        return full_adj

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

    def discover_structure(self, data: pd.DataFrame, model=None) -> Tuple[np.ndarray, List]:
        """Discover causal structure from X-only data.

        Parameters
        ----------
        data : pd.DataFrame
            Feature matrix **without** the target column Y.  The discovery
            algorithm runs purely on X features; edges to Y are added
            afterwards via :meth:`_build_full_adjacency`.
        model : optional
            Fitted predictive model (must expose ``selected_features``).  When
            provided its feature set determines which X nodes connect to Y.
            Sink nodes in the X-only graph are always connected to Y as well.

        Returns
        -------
        full_adjacency : np.ndarray  shape (n_x+1, n_x+1)
            Adjacency matrix covering X features **and** Y (last row/column).
        confounders : List[Tuple[str, str]]
        """
        print(" Running PC algorithm for causal structure discovery...")
        data_array = data.values
        feature_names = data.columns.tolist()  # X features only
        n_features = len(feature_names)

        indep_test_func = self._get_independence_test()

        try:
            # PC algorithm for causal discovery (X features only)
            self.pc_result = pc(
                data_array,
                alpha=self.alpha,
                indep_test=indep_test_func,
                stable=True,
                uc_rule=0,
                uc_priority=2
            )
            # Extract X-only adjacency matrix from PC result
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
            confounders = self._detect_confounders_from_graph(fci_graph, feature_names)

            print(f"FCI detected {len(confounders)} potential confounders pairs")
        
        except Exception as e:
            print(f"FCI algorithm failed: {str(e)}")
            confounders = []

        # Build full adjacency (X + Y), adding Y edges based on model + sinks
        full_adjacency = self._build_full_adjacency(adjacency_pc, feature_names, model)
        n_y_edges = int(full_adjacency[:n_features, n_features].sum())
        print(f"Y edges added: {n_y_edges} (model features ∪ sink nodes)")

        self.discovered_graph = full_adjacency
        self.confounders = confounders

        return full_adjacency, confounders
        
    def get_causal_relationships(self, data: pd.DataFrame, model=None) -> Dict:
        """Run discovery and return a results dict.

        Parameters
        ----------
        data : pd.DataFrame
            X-only feature matrix (no Y column).
        model : optional
            Fitted predictive model used to determine Y edges.
        """
        adjacency, confounders = self.discover_structure(data, model)

        results = {
            'method': 'PC + FCI',
            'adjacency_matrix': adjacency,
            'confounders': confounders,
            'feature_names': data.columns.tolist() + ['Y'],
            'n_edges': np.sum(adjacency != 0),
            'n_confounders_pairs': len(confounders)
        }

        return results

class LiNGAMWithFCI(CausalDiscoveryMethod):

    def __init__(self,alpha: float = 0.05, indep_test: str= 'fisherz'):

        super().__init__(alpha=alpha, indep_test=indep_test)
        self.lingam_result= None
        self.fci_result = None
    
    def discover_structure(self, data: pd.DataFrame, model=None) -> Tuple[np.ndarray, List]:
        """Discover causal structure from X-only data.

        Parameters
        ----------
        data : pd.DataFrame
            Feature matrix **without** the target column Y.  The discovery
            algorithm runs purely on X features; edges to Y are added
            afterwards via :meth:`_build_full_adjacency`.
        model : optional
            Fitted predictive model (must expose ``selected_features``).  When
            provided its feature set determines which X nodes connect to Y.
            Sink nodes in the X-only graph are always connected to Y as well.

        Returns
        -------
        full_adjacency : np.ndarray  shape (n_x+1, n_x+1)
            Adjacency matrix covering X features **and** Y (last row/column).
        confounders : List[Tuple[str, str]]
        """
        print("Running LiNGAM algorithm for causal structure discovery...")

        data_array = data.values
        feature_names = data.columns.tolist()  # X features only
        n_features = len(feature_names)

        # Run LiNGAM algorithm
        try:
            lingam_model = DirectLiNGAM()
            lingam_model.fit(data_array)

            # DirectLiNGAM convention: adjacency_matrix_[i, j] is the
            # coefficient of X_j *on* X_i, i.e. edge j -> i.
            # Our convention: adj[i, j] = 1 means edge i -> j.
            # Fix: transpose before binarising.
            adjacency_lingam = lingam_model.adjacency_matrix_.T  # now [i,j] = effect of X_i on X_j
            
            threashold = 0.1
            adjacency_lingam[np.abs(adjacency_lingam) < threashold] = 0

            adjacency_binary = (np.abs(adjacency_lingam) > 0).astype(int)

            print(f"LiNGAM discovered {np.sum(adjacency_binary != 0)} edges")

            self.lingam_result = lingam_model
        except Exception as e:
            print(f"LiNGAM algorithm failed: {str(e)}")
            print(f"Trying ICA-based LiNGAm as fallback...")
            try:
                # Fallback to ICA-based LiNGAM
                from causallearn.search.FCMBased.lingam import ICALiNGAM
                
                ica_model = ICALiNGAM()
                ica_model.fit(data_array)
                adjacency_lingam = ica_model.adjacency_matrix_
                
                threashold = 0.01
                adjacency_lingam[np.abs(adjacency_lingam) < threashold] = 0
                adjacency_binary = (np.abs(adjacency_lingam) > 0).astype(int)

                print(f"ICA-LiNGAM discovered {np.sum(adjacency_binary !=0 )} edges")
                self.lingam_result = ica_model

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
            confounders = self._detect_confounders_from_graph(fci_graph, feature_names)

            print(f"FCI detected {len(confounders)} potential confounders pairs")
        
        except Exception as e:
            print(f"FCI algorithm failed: {str(e)}")
            confounders = []

        # Build full adjacency (X + Y), adding Y edges based on model + sinks
        full_adjacency = self._build_full_adjacency(adjacency_binary, feature_names, model)
        n_y_edges = int(full_adjacency[:n_features, n_features].sum())
        print(f"Y edges added: {n_y_edges} (model features ∪ sink nodes)")

        self.discovered_graph = full_adjacency
        self.confounders = confounders

        return full_adjacency, confounders
    
    def get_causal_relationships(self, data: pd.DataFrame, model=None) -> Dict:
        """Run discovery and return a results dict.

        Parameters
        ----------
        data : pd.DataFrame
            X-only feature matrix (no Y column).
        model : optional
            Fitted predictive model used to determine Y edges.
        """
        adjacency, confounders = self.discover_structure(data, model)

        results = {
            'method': 'LiNGAM + FCI',
            'adjacency_matrix': adjacency,
            'confounders': confounders,
            'feature_names': data.columns.tolist() + ['Y'],
            'n_edges': np.sum(adjacency != 0),
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