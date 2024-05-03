from model import FastViTMLP, ResNET50L2CS, FastViTL2CS, FasterViT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import MPIIFaceGazeDataset, MPIIFaceGazeProcessedDataset
import utils
import timm
import torchvision
import torchvision.transforms as transforms


val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def validate(model, device, dataset,  test=False):
       
    # Load the model
    model.eval()

    angular_error = utils.AverageMeter()
    if dataset == "MPIIFaceGaze":
        from dataloader import MPIIFaceGazeDataset

        participans = [f"p{i:02d}" for i in range(15) if f"p{i:02d}"] 

        for participant in participans:
            angular_error.reset()
            dataset = MPIIFaceGazeProcessedDataset(
                "/home/kovan-beta/Desktop/visionmate/datasets/MPIIFaceGazeProcessed", participant, train=False, transform=val_transform
            )
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            len_loader = len(loader)
            i = 0
            for images,_,  gaze_directions in loader:
                with torch.no_grad():
                    images, gaze_directions = images.to(device), gaze_directions.to(device)
                    outputs = model(images)


                    angular_error.update(utils.compute_angular_error(outputs, gaze_directions), images.size(0))
                    i += images.size(0)
                    print(f"Participant: {participant} Batch Angular Error: {angular_error.val:.2f}", end="\r")
            print("" * 100,end="\r")
            print(f"Participant: {participant} Angular Error: {angular_error.avg:.2f}")

    else:
        from dataloader import Gaze360Dataset

        test_dataset = Gaze360Dataset(data_dir="/home/kovan-beta/Desktop/visionmate/datasets/Gaze360", file_name="test.txt", transform=val_transform)
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

        print("" * 100)
        print(f"Angular error: {angular_error.avg:.2f}")


def validate_l2cs(model, device, test=False, test_participant = "p00"):
        model.eval()

        softmax = nn.Softmax(dim=1).to(device)
        idx_tensor = torch.FloatTensor([idx for idx in range(28)]).to(device)
        angular_error = utils.AverageMeter()

        participans = [f"p{i:02d}" for i in range(15) if f"p{i:02d}" != test_participant] 

        for participant in participans:
            angular_error.reset()
            dataset = MPIIFaceGazeProcessedDataset(
                "/home/kovan-beta/Desktop/visionmate/datasets/MPIIFaceGazeProcessed", participant, train=False, transform=val_transform
            )
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            len_loader = len(loader)
            i = 0
            with torch.no_grad():
                for images, labels, cont_labels in loader:
                    images, labels, cont_labels = images.to(device), labels.to(device), cont_labels.to(device)

                    pre_yaw_gaze, pre_pitch_gaze = model(images)

                    # MSE loss
                    pitch_predicted = softmax(pre_pitch_gaze)
                    yaw_predicted = softmax(pre_yaw_gaze)

                    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    angular_error.update(utils.compute_angular_error(torch.stack((yaw_predicted, pitch_predicted), dim=1), cont_labels), images.size(0))
                    i += images.size(0)
                    print(f"Participant: {participant} Batch Angular Error: {angular_error.val:.2f}", end="\r")
            print("" * 100,end="\r")
            print(f"Participant: {participant} Angular Error: {angular_error.avg:.2f}")

if __name__ == "__main__":

    device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available else "cpu"
        )
    
    dataset = "MPIIFaceGaze"
    # model = ResNET50L2CS(device, 28, torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
    # model= nn.DataParallel(model,device_ids=[0])
    # saved_state_dict = torch.load("/home/kovan-beta/L2CS-Net/output/published_snapshots/fold0/fold0.pkl")
    # saved_state_dict = torch.load("/home/kovan-beta/L2CS-Net/output/resnet50_snapshots/fold0/_epoch_4.pkl")

    # model = FastViTL2CS(device, 28)
    # saved_state_dict = torch.load("/home/kovan-beta/L2CS-Net/output/fastvit36_snapshots/fold0/_epoch_60.pkl")
    
    model = FasterViT(device)
    saved_state_dict = torch.load("/home/kovan-beta/Desktop/visionmate/VisionMate/models/fastervit_gaze360-2.pth")

    model.load_state_dict(saved_state_dict["state_dict"])

    validate(model, device, dataset, test=True)

    