from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from torchvision import transforms, models
import utils
from model import FastViTMLP, FasterViT, FastViTL2CS, ResNET50L2CS, FastViTL2CS

class BaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gaze_directions = self.samples[idx]
        normalized_gaze = gaze_directions / np.linalg.norm(gaze_directions)
        spherical_vector = torch.FloatTensor(2)
        spherical_vector[0] = np.arctan2(normalized_gaze[0],-normalized_gaze[2])
        spherical_vector[1] = np.arcsin(normalized_gaze[1])

        # gaze_angles = torch.Tensor(gaze_directions)
        # gaze_angles = torch.FloatTensor(gaze_angles)
        # normalized_gaze = gaze_angles / np.linalg.norm(gaze_angles)
        # spherical_vector = torch.FloatTensor(2)
        # spherical_vector[0] = np.arctan2(normalized_gaze[0],-normalized_gaze[2])
        # spherical_vector[1] = np.arcsin(normalized_gaze[1])
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, spherical_vector
        ######################################################################3
        #         # Bin values
        # pitch = spherical_vector[0] * 180 / np.pi
        # yaw = spherical_vector[1]  * 180 / np.pi
        # bins = np.array(range(-42, 42, 3))
        # binned_pose = np.digitize([pitch, yaw], bins) - 1

        # labels = binned_pose
        # cont_labels = torch.FloatTensor([pitch, yaw])

        # return image, labels, cont_labels

class MPIIFaceGazeDataset(BaseDataset):
    def __init__(self, data_dir, test_participant, train, transform=None):
        super().__init__(data_dir, transform)
        if train:
            self.participants = [f"p{i:02d}" for i in range(15)]  # List participant folders
            self.participants.remove(test_participant)
        else:
            self.participants = [test_participant]

        for participant in self.participants:
            participant_path = os.path.join(self.data_dir, participant)
            with open(os.path.join(participant_path, f"{participant}.txt")) as f:
                lines = f.readlines()
                for line in lines:
                    info = line.split()
                    img_path = os.path.join(participant_path, info[0])
                    fc = [float(info[21]), float(info[22]), float(info[23])] #face center
                    gt = [float(info[24]), float(info[25]), float(info[26])] #gaze target point

                    gaze_direction = [gt[0] - fc[0], gt[1] - fc[1], gt[2] - fc[2]]

                    self.samples.append((img_path, gaze_direction))

class MPIIFaceGazeProcessedDataset(BaseDataset):
    def __init__(self, data_dir, test_participant, train, transform=None):
        super().__init__(data_dir, transform)
        if train:
            self.participants = [f"p{i:02d}" for i in range(15)]
            self.participants.remove(test_participant)
        else:
            self.participants = [test_participant]

        for participant in self.participants:
            participant_path = os.path.join(self.data_dir, "Label", participant + ".label")
            with open(participant_path) as f:
                lines = f.readlines()
                for line in lines[1:]:
                    info = line.strip().split()
                    img_path = os.path.join(self.data_dir, "Image", info[0])
                    gaze_direction = info[7].split(",")
                    gaze_direction = [float(gaze_direction[0]), float(gaze_direction[1])]
                    self.samples.append((img_path, gaze_direction))

    def __getitem__(self, idx):
        img_path, gaze_directions = self.samples[idx]
        gaze_angles = torch.Tensor(gaze_directions)
        gaze_angles = torch.FloatTensor(gaze_angles)

        pitch = gaze_angles[0] #* 180 / np.pi
        yaw = gaze_angles[1] #* 180 / np.pi

        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        # Bin values
        bins = np.array(range(-42, 42, 3))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])

        return img, labels, cont_labels

class Gaze360Dataset(BaseDataset):
    def __init__(self, data_dir, file_name, transform=None):
        super().__init__(data_dir, transform)
        self.file_name = file_name
        with open(os.path.join(self.data_dir, file_name)) as f:
            lines = f.readlines()
            for line in lines:
                info = line.split()
                img_path = os.path.join(self.data_dir + "/imgs/", info[0])
                gaze_direction = [float(info[1]), float(info[2]), float(info[3])]
                self.samples.append((img_path, gaze_direction))


if __name__ == "__main__":

    # Example usage
    data_dir = "/home/kovan-beta/Desktop/visionmate/datasets/Gaze360/"
    
    val_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = Gaze360Dataset(data_dir, "train.txt", transform=val_transform)
    test_dataset = Gaze360Dataset(data_dir, "test.txt", transform=val_transform)

    # dataset = MPIIFaceGazeDataset("/home/kovan-beta/Desktop/visionmate/datasets/MPIIFaceGaze", "p00", train=True, transform=None)
    # test_dataset = MPIIFaceGazeDataset("/home/kovan-beta/Desktop/visionmate/datasets/MPIIFaceGaze", "p00", train=False, transform=val_transform)    

    # dataset = MPIIFaceGazeProcessedDataset("/home/kovan-beta/Desktop/visionmate/datasets/MPIIFaceGazeProcessed", "p00", train=True, transform=val_transform)
    # test_dataset = MPIIFaceGazeProcessedDataset("/home/kovan-beta/Desktop/visionmate/datasets/MPIIFaceGazeProcessed", "p00", train=False, transform=val_transform)

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model1 = FasterViT("cuda")
    state = torch.load("/home/kovan-beta/Desktop/visionmate/VisionMate/models/fastervit_gaze360-2.pth")
    model1.load_state_dict(state["state_dict"])
    model1.eval()

    model2 = ResNET50L2CS("cuda", 90, models.resnet.Bottleneck, [3, 4, 6, 3])
    model2.load_state_dict(torch.load("/home/kovan-beta/L2CS-Net/models/L2CSNet_gaze360.pkl"))
    model2.eval()

    for images, gaze_directions in test_dataloader:

        images_cuda, gaze_directions_cuda = images.to("cuda"), gaze_directions.to("cuda")

        if isinstance(model1, ResNET50L2CS):
            #resize the image to 448
            model1_pred = utils.L2CS_final_output(model1, images_cuda)
            model1_pred  = model1_pred[0][0], model1_pred[1][0]
        else:
            images_cuda = torch.nn.functional.interpolate(images_cuda, size=(224,224), mode="bilinear")
            outputs = model1(images_cuda)
            model1_pred = outputs[:, 0].item(), outputs[:, 1].item()

        
        if isinstance(model2, ResNET50L2CS):
            #resize the image to 448
            # images_cuda = torch.nn.functional.interpolate(images_cuda, size=(448,448), mode="bilinear")
            model2_pred = utils.L2CS_final_output(model2, images_cuda)
            model2_pred  = model2_pred[0][0], model2_pred[1][0]
            print(model2_pred)
        else:
            outputs = model2(images_cuda)
            model2_pred = outputs[:, 0].item(), outputs[:, 1].item()

        #display the image in opencv and draw the gaze direction
        image = images[0].permute(1, 2, 0).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gaze = gaze_directions[0].numpy()

        #resize the image to 448
        image = cv2.resize(image, (448, 448))

        x1, y1, x2, y2 = 0, 0, image.shape[1], image.shape[0]

        pitch = gaze[0]
        yaw = gaze[1]
        utils.draw_gaze(x1, y1, x2, y2, image, model1_pred, color=(0, 0, 255)) 
        utils.draw_gaze(x1, y1, x2, y2, image, model2_pred, color=(255, 0, 0))
        utils.draw_gaze(x1, y1, x2, y2, image, gaze, color=(0, 255, 0))


        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()