import numpy as np
import torch


# Check if MPS is available and set the device to 'mps', otherwise fallback to 'cpu'
def set_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


# ------ Action metrics
def compute_action_metrics(actions_batch):
    """
    Compute various metrics about the actions to understand policy behavior.
    actions_batch: numpy array of shape (batch_size, 2) where columns are [A1, A2]
    """
    actions_tensor = torch.FloatTensor(actions_batch)

    # Separate A1 and A2
    a1_values = actions_tensor[:, 0]
    a2_values = actions_tensor[:, 1]

    metrics = {
        # Mean absolute values - if A1 is irrelevant, it might converge to smaller values
        "mean_abs_a1": torch.mean(torch.abs(a1_values)).item(),
        "mean_abs_a2": torch.mean(torch.abs(a2_values)).item(),

        # Standard deviations - irrelevant actions might have less variance
        "std_a1": torch.std(a1_values).item(),
        "std_a2": torch.std(a2_values).item(),

        # Ratio of action magnitudes
        "action_magnitude_ratio": (torch.mean(torch.abs(a2_values)) /
                                   (torch.mean(torch.abs(a1_values)) + 1e-8)).item(),

        # How often each action saturates (hits ±1)
        "a1_saturation_rate": (torch.sum(torch.abs(a1_values) > 0.95) / len(a1_values)).item(),
        "a2_saturation_rate": (torch.sum(torch.abs(a2_values) > 0.95) / len(a2_values)).item(),

        # Energy/effort metrics
        "a1_energy": torch.mean(a1_values ** 2).item(),
        "a2_energy": torch.mean(a2_values ** 2).item(),
    }

    return metrics


@torch.no_grad()  # This decorator ensures no gradient tracking
def analyze_policy_gradients(states_batch, policy_agent):
    """
    Analyze how sensitive the policy is to each action dimension.
    This tells us which actions the policy is actively trying to optimize.
    """
    # Create a completely fresh copy of states, detached from any computation graph
    states_tensor = torch.FloatTensor(np.array(states_batch)).to(policy_agent.device)

    # Enable gradient tracking for the states
    with torch.enable_grad():
        # Create a fresh tensor that requires gradients
        states_for_analysis = states_tensor.detach().requires_grad_(True)

        # Get actions from the policy - this creates a new computation graph
        mean, log_std = policy_agent.actor(states_for_analysis)

        # Compute gradient magnitudes for each action dimension
        grad_metrics = {}

        for i, action_name in enumerate(['a1', 'a2']):
            # Compute gradients without affecting parameter gradients
            grad = torch.autograd.grad(
                outputs=mean[:, i].sum(),
                inputs=states_for_analysis,  # Only compute w.r.t. our input
                retain_graph=True,  # Keep graph for next iteration
                create_graph=False,  # Don't create higher-order gradient graph
                only_inputs=True  # Extra safety: only compute input gradients
            )[0]

            grad_magnitude = torch.norm(grad, dim=1).mean()
            grad_metrics[f'policy_gradient_magnitude_{action_name}'] = grad_magnitude.item()

        # At this point, the temporary computation graph is destroyed
        # and cannot affect any training gradients

    return grad_metrics
