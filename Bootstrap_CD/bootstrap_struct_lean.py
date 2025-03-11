import numpy as np
import pandas as pd

# Import causal-learn algorithms.
# (Make sure you have installed causal-learn via pip: pip install causal-learn)
from causallearn.search.ConstraintBased import PC, FCI
from causallearn.search.ScoreBased import GES

# For reproducibility
np.random.seed(42)


def enforce_edge_prior(edge_dict, edge_priors, n_bootstrap):
    """
    Force a priori edges into the aggregated edge frequency dictionary.

    Parameters
    ----------
    edge_dict : dict
        Aggregated edge frequency dictionary from bootstrap runs.
    edge_priors : list of tuples
        List of edges (u, v) that must be present.
    n_bootstrap : int
        Number of bootstrap runs (to assign a maximal frequency).

    Returns
    -------
    dict
        Updated aggregated edge frequency dictionary.
    """
    for edge in edge_priors:
        # Force the edge frequency to be maximal
        edge_dict[edge] = n_bootstrap
    return edge_dict


class BootstrapStructureLearner:
    """
    Class to run structure learning with bootstrapping (either standard or Bayesian bootstrap)
    using a chosen algorithm (PC, FCI, or GES) from the causal-learn package.
    """

    def __init__(self, algorithm='PC', n_bootstrap=50, bootstrap_type='standard', edge_priors=None):
        """
        Parameters
        ----------
        algorithm : str
            One of 'PC', 'FCI', or 'GES'.
        n_bootstrap : int
            Number of bootstrap replicates.
        bootstrap_type : str
            'standard' for classical bootstrap or 'bayesian' for Bayesian bootstrap.
        edge_priors : list of tuples or None
            List of edges to force (e.g. [(0, 1)]).
        """
        self.algorithm = algorithm
        self.n_bootstrap = n_bootstrap
        self.bootstrap_type = bootstrap_type.lower()
        self.edge_priors = edge_priors

    def run_structure_learning(self, data):
        """
        Run the chosen structure learning algorithm on the provided data.

        Parameters
        ----------
        data : np.ndarray
            2D numpy array where each column is a variable.

        Returns
        -------
        learner : object
            The output object from the causal-learn algorithm, assumed to have a .G attribute.
        """
        if self.algorithm == 'PC':
            learner = PC(data, alpha=0.05)
        elif self.algorithm == 'FCI':
            learner = FCI(data, alpha=0.05)
        elif self.algorithm == 'GES':
            learner = GES(data)
        else:
            raise ValueError("Unsupported algorithm: choose 'PC', 'FCI', or 'GES'")
        return learner

    def extract_edges(self, learner):
        """
        Extract directed edges from the learner's output.
        Assumes learner.G is a graph-like object with an 'edges' attribute.

        Parameters
        ----------
        learner : object
            Output from a causal-learn algorithm.

        Returns
        -------
        list of tuples
            List of edges represented as (u, v).
        """
        # Note: Depending on the algorithm, learner.G may be a networkx graph.
        # Here we assume that learner.G.edges is iterable over (u,v) tuples.
        return list(learner.G.edges)

    def bootstrap(self, data):
        """
        Run bootstrap replicates of structure learning on the given data and aggregate edge frequencies.

        Parameters
        ----------
        data : np.ndarray
            2D numpy array of shape (n_samples, n_features).

        Returns
        -------
        dict
            Dictionary with keys as edges (tuples) and values as frequency counts.
        """
        aggregated_edges = {}
        n_samples = data.shape[0]
        for i in range(self.n_bootstrap):
            # Create a bootstrap sample.
            if self.bootstrap_type == 'standard':
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            elif self.bootstrap_type == 'bayesian':
                # Bayesian bootstrap: draw weights from a Dirichlet distribution
                weights = np.random.dirichlet(np.ones(n_samples))
                # Sample indices proportional to the weights (still sampling n_samples times)
                indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights)
            else:
                raise ValueError("Unsupported bootstrap type. Choose 'standard' or 'bayesian'.")

            data_sample = data[indices]
            learner = self.run_structure_learning(data_sample)
            edges = self.extract_edges(learner)
            for edge in edges:
                aggregated_edges[edge] = aggregated_edges.get(edge, 0) + 1

        # Enforce a priori domain knowledge (if provided)
        if self.edge_priors is not None:
            aggregated_edges = enforce_edge_prior(aggregated_edges, self.edge_priors, self.n_bootstrap)
        return aggregated_edges


def main():
    # -------------------------------
    # Create a simple synthetic dataset.
    # Suppose we have five variables with a known underlying structure.
    # Let columns correspond to variables A, B, C, D, and E.
    # For example, assume A influences B and C, B and C influence D, and D influences E.
    n_samples = 500
    A = np.random.normal(0, 1, n_samples)
    B = 2 * A + np.random.normal(0, 1, n_samples)
    C = -A + np.random.normal(0, 1, n_samples)
    D = 0.5 * B + 0.3 * C + np.random.normal(0, 1, n_samples)
    E = D + np.random.normal(0, 1, n_samples)

    # Combine variables into a numpy array.
    # Variable order: 0: A, 1: B, 2: C, 3: D, 4: E.
    data = np.column_stack([A, B, C, D, E])

    # Define a prior edge.
    # For instance, suppose we know from domain knowledge that A → B must exist.
    # Since A is column 0 and B is column 1, we define:
    edge_priors = [(0, 1)]

    # -------------------------------
    # Run structure learning using different algorithms with bootstrap.

    # Example using the PC algorithm with standard bootstrap.
    bsl_pc = BootstrapStructureLearner(algorithm='PC', n_bootstrap=20,
                                       bootstrap_type='standard', edge_priors=edge_priors)
    aggregated_edges_pc = bsl_pc.bootstrap(data)
    print("Aggregated edge frequencies (PC, standard bootstrap):")
    for edge, freq in aggregated_edges_pc.items():
        print(f"Edge {edge}: Frequency {freq}")

    # Example using the GES algorithm with Bayesian bootstrap.
    bsl_ges = BootstrapStructureLearner(algorithm='GES', n_bootstrap=20,
                                        bootstrap_type='bayesian', edge_priors=edge_priors)
    aggregated_edges_ges = bsl_ges.bootstrap(data)
    print("\nAggregated edge frequencies (GES, Bayesian bootstrap):")
    for edge, freq in aggregated_edges_ges.items():
        print(f"Edge {edge}: Frequency {freq}")

    # Optionally, you could also run FCI in the same way:
    bsl_fci = BootstrapStructureLearner(algorithm='FCI', n_bootstrap=20,
                                        bootstrap_type='standard', edge_priors=edge_priors)
    aggregated_edges_fci = bsl_fci.bootstrap(data)
    print("\nAggregated edge frequencies (FCI, standard bootstrap):")
    for edge, freq in aggregated_edges_fci.items():
        print(f"Edge {edge}: Frequency {freq}")


if __name__ == '__main__':
    main()
