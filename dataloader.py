from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
from torchvision import transforms

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
        # return image, spherical_vector
        ######################################################################3
                # Bin values
        pitch = spherical_vector[0] * 180 / np.pi
        yaw = spherical_vector[1]  * 180 / np.pi
        bins = np.array(range(-42, 42, 3))
        binned_pose = np.digitize([pitch, yaw], bins) - 1

        labels = binned_pose
        cont_labels = torch.FloatTensor([pitch, yaw])

        return image, labels, cont_labels

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

        pitch = gaze_angles[0] * 180 / np.pi
        yaw = gaze_angles[1] * 180 / np.pi

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
    data_dir = "./data/MPIIFaceGaze"
    dataset = MPIIFaceGazeDataset(data_dir)

    # Split dataset into train and test sets
    # train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    # print(len(train_dataset))  # Output: 1200
    # print(len(test_dataset))  # Output: 300

    # Create dataloaders for train and test sets
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for images, gaze_directions in train_dataloader:
        print("Train:", images.shape, gaze_directions.shape)
        break

    # for images, gaze_directions in test_dataloader:
    #     print("Test:", images.shape, gaze_directions.shape)
    #     break

