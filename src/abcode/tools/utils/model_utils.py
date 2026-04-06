import torch


def iter_batches(items, batch_size):
    """
    Yield fixed-size slices from a sequence-like object.
    """
    size = max(int(batch_size), 1)
    for start in range(0, len(items), size):
        yield items[start : start + size]

def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    else:
        device = 'cpu'
    print('Device:', device)
    return device
