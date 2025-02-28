import torch


# Check if MPS is available and set the device to 'mps', otherwise fallback to 'cpu'
def set_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device