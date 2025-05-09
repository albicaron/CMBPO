import numpy as np
import matplotlib.pyplot as plt

from castle.common import GraphDAG
from castle.common.priori_knowledge import PrioriKnowledge
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import PC, DirectLiNGAM


def simulate_data(n_node=10, n_edge=10, n=1000, method='linear', sem_type='gauss'):
    weighted_random_dag = DAG.erdos_renyi(n_nodes=n_node, n_edges=n_edge,
                                          weight_range=(0.3, 0.5), seed=1)
    dataset = IIDSimulation(W=weighted_random_dag, n=n, method=method,
                            sem_type=sem_type)
    true_causal_matrix, X = dataset.B, dataset.X
    return true_causal_matrix, X


def set_p_matrix(s_dim, a_dim):
    tot_nodes = s_dim + a_dim + s_dim + 1  # (s,a,ns,r)
    p_matrix = np.zeros((tot_nodes, tot_nodes)) - 1  # Initialize all to learnable
    np.fill_diagonal(p_matrix, 0)  # No self-edges

    # Index ranges for (s,a,ns,r)
    s_idx = np.arange(0, s_dim)
    a_idx = np.arange(s_dim, s_dim + a_dim)
    ns_idx = np.arange(s_dim + a_dim, s_dim + a_dim + s_dim)
    r_idx = np.array([tot_nodes - 1])

    # Set forbidden edges (0s)

    # No s to a, or a to s (no intra-layer edges between s and a)
    p_matrix[np.ix_(s_idx, a_idx)] = 0
    p_matrix[np.ix_(a_idx, s_idx)] = 0

    # No edges to s or a (from ns or r or anywhere)
    p_matrix[:, s_idx] = 0
    p_matrix[:, a_idx] = 0

    # No edges from ns or r to anywhere (no reverse time edges)
    p_matrix[ns_idx, :] = 0
    p_matrix[r_idx, :] = 0

    # Allow only learnable edges from (s, a) to (ns, r)
    learnable_from = np.concatenate([s_idx, a_idx])
    learnable_to = np.concatenate([ns_idx, r_idx])
    # We already initialized everything to -1, so make everything else 0
    p_matrix[:, :] = 0
    p_matrix[np.ix_(learnable_from, learnable_to)] = -1
    np.fill_diagonal(p_matrix, 0)

    return p_matrix


class StructureLearning:
    def __init__(self, n_nodes, sl_method="PC", bootstrap=None):
        self.n_nodes = n_nodes
        self.sl_method = sl_method

        # check the bootstrap either is None, 'standard', or 'bayesian'
        if bootstrap not in [None, 'standard', 'bayesian']:
            raise ValueError("The bootstrap should be either None, 'standard', or 'bayesian'.")

        self.bootstrap = bootstrap

    def learn_dag(self, X, prior_knowledge=None, n_bootstrap=20):

        if self.sl_method == "PC":
            causal_matrix = self.pc_learn(X, prior_knowledge, n_bootstrap)
        elif self.sl_method == "LiNGAM":
            causal_matrix = self.lingam_learn(X, prior_knowledge, n_bootstrap)
        else:
            raise ValueError("The structure learning method is not supported.")

        return causal_matrix

    def set_prior_knowledge(self, p_matrix):
        p = PrioriKnowledge(self.n_nodes)

        # Based on p_matrix, add forbidden edges according to the prior p_matrix
        no_edge_constr = []
        for i in range(p_matrix.shape[0]):
            for j in range(p_matrix.shape[1]):
                if p_matrix[i, j] == 0:
                    no_edge_constr.append((i, j))

        p.add_forbidden_edges(no_edge_constr)

        return p

    def pc_learn(self, X, p=None, n_bootstrap=20):

        if self.bootstrap == 'standard':

            aggregated_matrix = np.zeros((self.n_nodes, self.n_nodes))
            n_samples = X.shape[0]

            for i in range(n_bootstrap):
                # Resample with replacement (classical bootstrap)
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X[indices, :]

                # Learn graph on bootstrap sample with the prior knowledge
                pc = PC(priori_knowledge=p)
                pc.learn(X_boot)

                # Aggregate (here simply summing the binary matrices)
                aggregated_matrix += pc.causal_matrix

            # Average over replicates to get edge frequencies (0 to 1)
            aggregated_matrix /= n_bootstrap

            causal_matrix = aggregated_matrix

        elif self.bootstrap == 'bayesian':
            aggregated_matrix = np.zeros((self.n_nodes, self.n_nodes))
            n_samples = X.shape[0]

            for i in range(n_bootstrap):
                # Draw weights from a Dirichlet distribution (each weight is non-negative and sums to 1)
                weights = np.random.dirichlet(np.ones(n_samples))

                # Use weights as probabilities to sample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights)
                X_bayes = X[indices, :]

                # Learn graph on Bayes bootstrap sample with the prior knowledge
                pc = PC(priori_knowledge=p, alpha=0.01)
                pc.learn(X_bayes)

                # Aggregate (here simply summing the binary matrices)
                aggregated_matrix += pc.causal_matrix

            # Average over replicates to get edge frequencies (0 to 1)
            aggregated_matrix /= n_bootstrap
            causal_matrix = aggregated_matrix

        else:
            # No bootstrap
            pc = PC(priori_knowledge=p)
            pc.learn(X)
            causal_matrix = pc.causal_matrix

        return causal_matrix

    def lingam_learn(self, X, p=None, n_bootstrap=20):

        if self.bootstrap == 'standard':

            aggregated_matrix = np.zeros((self.n_nodes, self.n_nodes))
            n_samples = X.shape[0]

            for i in range(n_bootstrap):
                # Resample with replacement (classical bootstrap)
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X[indices, :]

                # Learn graph on bootstrap sample with the prior knowledge
                lingam = DirectLiNGAM(prior_knowledge=p.matrix)
                lingam.learn(X_boot)

                # Aggregate (here simply summing the binary matrices)
                aggregated_matrix += lingam.causal_matrix

            # Average over replicates to get edge frequencies (0 to 1)
            aggregated_matrix /= n_bootstrap

            causal_matrix = aggregated_matrix

        elif self.bootstrap == 'bayesian':

            aggregated_matrix = np.zeros((self.n_nodes, self.n_nodes))
            n_samples = X.shape[0]

            for i in range(n_bootstrap):
                # Draw weights from a Dirichlet distribution (each weight is non-negative and sums to 1)
                weights = np.random.dirichlet(np.ones(n_samples))

                # Use weights as probabilities to sample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights)
                X_bayes = X[indices, :]

                # Learn graph on Bayes bootstrap sample with the prior knowledge
                lingam = DirectLiNGAM(prior_knowledge=p.matrix)
                lingam.learn(X_bayes)

                # Aggregate (here simply summing the binary matrices)
                aggregated_matrix += lingam.causal_matrix

            # Average over replicates to get edge frequencies (0 to 1)
            aggregated_matrix /= n_bootstrap
            causal_matrix = aggregated_matrix

        else:
            # No bootstrap
            try:
                lingam = DirectLiNGAM(prior_knowledge=p.matrix)
            except AttributeError:
                lingam = DirectLiNGAM()
            lingam.learn(X)
            causal_matrix = lingam.causal_matrix

        return causal_matrix

    def sample_causal_mask(self, parent_set_proba):
        causal_mask = np.random.binomial(1, parent_set_proba, size=(self.n_nodes, self.n_nodes))
        np.fill_diagonal(causal_mask, 0)
        return causal_mask

    def plot_dag(self, predict_dag, true_dag, save_name=None, labels=None, figsize=(12, 5), cmap_true='Blues',
                 cmap_pred='Oranges'):
        """
        Plot predicted and true DAGs as adjacency matrices side by side using Matplotlib only.

        Args:
            predict_dag (np.ndarray): Predicted DAG adjacency matrix.
            true_dag (np.ndarray): Ground truth DAG adjacency matrix.
            save_name (str): Optional filename to save the plot.
            labels (list of str): Optional list of labels for the axes.
            figsize (tuple): Size of the full figure.
            cmap_true (str): Colormap for the true DAG.
            cmap_pred (str): Colormap for the predicted DAG.
        """

        if true_dag is not None:

            assert predict_dag.shape == true_dag.shape, "predict_dag and true_dag must have the same shape"
            n = predict_dag.shape[0]
            vmax = max(np.max(predict_dag), np.max(true_dag), 1)

            fig, axes = plt.subplots(1, 2, figsize=figsize)

            # Plot true DAG
            im1 = axes[0].imshow(true_dag, cmap=cmap_true, vmin=0, vmax=vmax)
            axes[0].set_title("True DAG")
            axes[0].set_xticks(range(n))
            axes[0].set_yticks(range(n))
            if labels:
                axes[0].set_xticklabels(labels, rotation=90)
                axes[0].set_yticklabels(labels)
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            # Plot predicted DAG
            im2 = axes[1].imshow(predict_dag, cmap=cmap_pred, vmin=0, vmax=vmax)
            axes[1].set_title("Predicted DAG")
            axes[1].set_xticks(range(n))
            axes[1].set_yticks(range(n))
            if labels:
                axes[1].set_xticklabels(labels, rotation=90)
                axes[1].set_yticklabels(labels)
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

            plt.tight_layout()

        else:

            # If no true DAG plot only the predicted DAG
            n = predict_dag.shape[0]
            vmax = max(np.max(predict_dag), np.max(predict_dag), 1)

            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(predict_dag, cmap=cmap_pred, vmin=0, vmax=vmax)
            ax.set_title("Predicted DAG")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            if labels:
                ax.set_xticklabels(labels, rotation=90)
                ax.set_yticklabels(labels)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        return fig

    def calculate_metrics(self, predict_dag, true_dag):
        mt = MetricsDAG(predict_dag, true_dag)
        return mt.metrics



