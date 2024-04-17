import torch

def spherical2cartesial(x):

    output = torch.zeros(x.size(0), 3)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])

    return output

def compute_angular_error(input, target):

    #input shape: (batch_size, 2)
    #target shape: (batch_size, 2)

    # return the angle between two vectors
    norm_input = torch.sqrt(torch.sum(input ** 2, dim=1))
    norm_target = torch.sqrt(torch.sum(target ** 2, dim=1))
    dot_product = torch.sum(input * target, dim=1)
    cosine_similarity = dot_product / (norm_input * norm_target)
    cosine_similarity = torch.clamp(cosine_similarity, -1, 1)  # Clamp the cosine similarity to the valid range
    angle = torch.acos(cosine_similarity)
    mean_angle = torch.mean(angle) * 180 / 3.1415926

    return mean_angle


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMAMeter(object):
    """Computes the Exponential Moving Average of a value"""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.val = 0
        self.avg = 0

    def update(self, val):
        self.val = val
        self.avg = self.alpha * self.avg + (1 - self.alpha) * self.val