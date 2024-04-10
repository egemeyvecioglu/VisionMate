import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import matplotlib.pyplot as plt
from dataloader import MPIIFaceGazeDataset, Gaze360Dataset

# from dataloader import train_test_split
from model import FastViTMLP
from torch.utils.data import random_split
import os


class VisionMate:
    def __init__(self) -> None:
        self.num_epochs = 100
        self.batch_size = 64
        self.learning_rate = 3e-4
        self.model_path = "model.pth"
        self.dataset = "MPIIFaceGaze"
        self.data_dir = "./data/MPIIFaceGaze" if self.dataset == "MPIIFaceGaze" else "./data/Gaze360"
        self.train_losses = []
        self.val_losses = []
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available else "cpu"
        )
        self.current_epoch = 0

        self.state = None

        self.model = FastViTMLP(self.device)
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-5
        )  # cosine annealing learning rate

        # if there is a model.pth fil in the path, load it
        if os.path.exists(self.model_path):
            print("Loading from checkpoint")
            self.state = torch.load(self.model_path)
            self.model.load_state_dict(self.state["state_dict"])
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.scheduler.load_state_dict(self.state["scheduler"])
            self.train_losses = self.state["train_losses"]
            self.val_losses = self.state["val_losses"]
            epoch = self.state["epoch"]
            self.num_epochs = self.num_epochs - epoch

        self.train_loader, self.val_loader, self.test_loader = self.load_dataset()

    def load_dataset(self):
        # Load dataset
        data_config = timm.data.resolve_model_data_config(self.model.fastvit)
        transform = timm.data.create_transform(**data_config, is_training=True)
        if self.dataset == "MPIIFaceGaze":
            dataset = MPIIFaceGazeDataset(self.data_dir, transform)

            # Split dataset into train and test sets
            print("Splitting the dataset")
            train_size = int(0.95 * len(dataset))  # 80% of the data for training
            test_size = len(dataset) - train_size  # Remaining 20% for testing
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            val_dataset = None

        else: #Gaze360
            train_dataset = Gaze360Dataset(self.data_dir, "train.txt", transform)
            test_dataset = Gaze360Dataset(self.data_dir, "test.txt", transform)
            val_dataset = Gaze360Dataset(self.data_dir, "validation.txt", transform)

        print("Creating train loader")
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        print("Creating test loader")
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        print("Creating validation loader")
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset else None

        return train_dataloader, val_dataloader, test_dataloader

    def train(self):
        print("Training starts. Device: ", self.device)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch} learning rate: {self.scheduler.get_last_lr()[-1]}")
            self.model.train()
            train_loss = 0.0
            total_train_batches = len(self.train_loader)
            i = 0

            train_angular_error = AverageMeter()
            for images, gaze_directions in self.train_loader:
                images, gaze_directions = images.to(self.device), gaze_directions.to(
                    self.device
                )
                self.optimizer.zero_grad()
                outputs = self.model(images)
                train_angular_error.update(
                    self.compute_angular_error(outputs, gaze_directions).item(),
                    images.size(0),
                )
                loss = self.criterion(outputs, gaze_directions)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                print(f"Train: {i}/{len(self.train_loader.dataset)}, Batch Angular Error: {train_angular_error.val}", end="\r")
                i += images.size(0)

            self.scheduler.step()

            self.train_losses.append(train_loss / total_train_batches)

            print()

            val_loss, val_angular_error = self.validate(epoch)

            print(
                f"Epoch: {epoch}, Train Loss: {train_loss / total_train_batches}, Val Loss: {val_loss}, \
                    Train Angular Error: {train_angular_error.avg}, Val Angular Error: {val_angular_error}"
            )

            
    def validate(self, test=False):
        self.model.eval()
        val_angular_error = AverageMeter()
        val_loss = 0.0
        loader = self.test_loader if test or self.val_loader is None else self.val_loader
        len_loader = len(loader)
        i = 0
        val_angular_error.reset()
        with torch.no_grad():
            for images, gaze_directions in loader:
                images, gaze_directions = images.to(self.device), gaze_directions.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, gaze_directions)
                val_loss += loss.item()
                val_angular_error.update(
                    self.compute_angular_error(outputs, gaze_directions).item(),
                    images.size(0),
                )
                print(f"Val: {i}/{len(loader.dataset)}, Batch Angular Error: {val_angular_error.val}", end="\r")
                i += images.size(0)
        print()

        if not test:
            self.val_losses.append(val_loss / len_loader)

        return val_loss / len_loader, val_angular_error.avg


    def plot_losses(self):
        # plotting the loss
        plt.figure()
        plt.plot(self.train_losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("training_loss.png")

        # plot the accuracy
        plt.figure()
        plt.plot(self.val_losses)
        plt.title("Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("val_loss.png")

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt or Exception as e:
            print("Error: ", e)
        finally:
            self.state = {
                "epoch": self.current_epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
            }
            torch.save(self.state, self.model_path)

            self.plot_losses()

            if self.val_loader:
                test_loss, test_angular_error = self.validate(test=True)
                print(f"Test Loss: {test_loss}, Test Angular Error: {test_angular_error}")

            

    def spherical2cartesial(self, x):

        output = torch.zeros(x.size(0), 3)
        output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
        output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
        output[:, 1] = torch.sin(x[:, 1])

        return output

    def compute_angular_error(self, input, target):

        input = self.spherical2cartesial(input)
        target = self.spherical2cartesial(target)

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


if __name__ == "__main__":
    vm = VisionMate()
    vm.run()
