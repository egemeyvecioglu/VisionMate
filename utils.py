import torch

def spherical2cartesial(x):

    output = torch.zeros(x.size(0), 3)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])

    return output

def compute_angular_error(input, target):

    input = spherical2cartesial(input)
    target = spherical2cartesial(target)

    input = input.view(-1, 3, 1)
    target = target.view(-1, 1, 3)
    output_dot = torch.bmm(target, input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180 * torch.mean(output_dot) / torch.pi
    return output_dot


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