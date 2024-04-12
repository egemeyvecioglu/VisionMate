from model import FastViTMLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import MPIIFaceGazeDataset
import utils
import timm


if __name__ == "__main__":

    device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available else "cpu"
        )

    model_name = "model.pth"
    dataset = "MPIIFaceGaze"

    data_dir = (
        "./data/MPIIFaceGaze"
        if dataset == "MPIIFaceGaze"
        else "/home/kovan-beta/gaze360/Gaze360/"
    )

    # Load the model
    model = FastViTMLP(device)
    state = torch.load(model_name)
    model.load_state_dict(state["state_dict"])
    model.eval()

    data_config = timm.data.resolve_model_data_config(model.fastvit)
    transform = timm.data.create_transform(**data_config, is_training=True)

    if dataset == "MPIIFaceGaze":
        from dataloader import MPIIFaceGazeDataset
        test_participant = state["test_participant"]

        test_dataset = MPIIFaceGazeDataset(data_dir=data_dir, test_participant="p00", train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    else:
        from dataloader import Gaze360Dataset

        test_dataset = Gaze360Dataset(data_dir=data_dir, file_name="test.txt", transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    angular_error = utils.AverageMeter()
    i = 0
    for images, gaze_directions in test_loader:
        with torch.no_grad():
            images, gaze_directions = images.to(device), gaze_directions.to(device)
            outputs = model(images)
            angular_error.update(utils.compute_angular_error(outputs, gaze_directions), images.size(0))
            print(f"{i} / {len(test_loader.dataset)} Batch  Angular error: {angular_error.val:.2f}", end="\r")
            i += 64

    print(f"Angular error: {angular_error.avg:.2f}")

