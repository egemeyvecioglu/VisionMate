import torch
import numpy as np
import cv2

def L2CS_final_output(model, image):

    gaze_pitch, gaze_yaw = model(image)

    pitch_predicted = model.softmax(gaze_pitch)
    yaw_predicted = model.softmax(gaze_yaw)
    
    # Get continuous predictions in degrees.
    pitch_predicted = torch.sum(pitch_predicted.data * model.idx_tensor, dim=1) * 4 - 180
    yaw_predicted = torch.sum(yaw_predicted.data * model.idx_tensor, dim=1) * 4 - 180
    
    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

    return pitch_predicted, yaw_predicted


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

def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = c
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

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