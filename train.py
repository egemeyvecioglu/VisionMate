import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import matplotlib.pyplot as plt
from dataloader import MPIIFaceGazeDataset, Gaze360Dataset
import wandb
from model import FastViTMLP
from torch.utils.data import random_split
import os
import utils

class VisionMate:
    def __init__(self) -> None:
        self.num_epochs = 75
        self.batch_size = 16
        self.learning_rate = 1e-5
        self.model_path = "mpii.pth"
        self.dataset = "MPIIFaceGaze"
        self.data_dir = (
            "./data/MPIIFaceGaze"
            if self.dataset == "MPIIFaceGaze"
            else "/home/kovan-beta/gaze360/Gaze360/"
        )
        self.train_losses = []
        self.val_losses = []
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available else "cpu"
        )
        self.current_epoch = 0

        self.state = None
        # torch.set_float32_matmul_precision('high')
        self.model = FastViTMLP(self.device)
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4
        # )

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1, eta_min=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer, T_max=self.num_epochs, eta_min=1e-5
        # )  # cosine annealing learning rate

        self.test_participant = None

        # if there is a model.pth fil in the path, load it
        if os.path.exists(self.model_path):
            print("Loading from checkpoint")
            self.state = torch.load(self.model_path)
            self.model.load_state_dict(self.state["state_dict"])
            self.optimizer.load_state_dict(self.state["optimizer"])
            # self.scheduler.load_state_dict(self.state["scheduler"])
            self.train_losses = self.state["train_losses"]
            self.val_losses = self.state["val_losses"]
            epoch = self.state["epoch"]
            self.test_participant = self.state["test_participant"]
            self.num_epochs = self.num_epochs - epoch

        self.train_loader, self.val_loader, self.test_loader = self.load_dataset()

        wandb.init(
            project="VisionMate",
            config={
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
            },
        )  # Replace with your details

    def load_dataset(self):
        # Load dataset
        data_config = timm.data.resolve_model_data_config(self.model.fastvit)
        # transform = timm.data.create_transform(**data_config, is_training=True)

        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=256,scale=(0.8,1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        val_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.dataset == "MPIIFaceGaze":

            # randomly leave one participant out for testing
            randint = torch.randint(0, 15, (1,))
            self.test_participant = f"p{randint.item():02d}"

            print("Test participant: ", self.test_participant)

            train_dataset = MPIIFaceGazeDataset(
                self.data_dir, self.test_participant, train=True, transform=train_transform
            )
            test_dataset = MPIIFaceGazeDataset(
                self.data_dir, self.test_participant, train=False, transform=val_transform
            )
            val_dataset = None

        else:  # Gaze360
            train_dataset = Gaze360Dataset(self.data_dir, "train.txt", train_transform)
            test_dataset = Gaze360Dataset(self.data_dir, "test.txt", val_transform)
            val_dataset = Gaze360Dataset(self.data_dir, "validation.txt", val_transform)

        print("Creating train loader")
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        print("Creating test loader")
        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        print("Creating validation loader")
        val_dataloader = (
            DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            if val_dataset
            else None
        )

        return train_dataloader, val_dataloader, test_dataloader

    def train(self):
        print("Training starts. Device: ", self.device)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            # print(f"Epoch {epoch} learning rate: {self.scheduler.get_last_lr()[-1]}")
            self.model.train()
            train_loss = 0.0
            total_train_batches = len(self.train_loader)
            i = 0

            train_angular_error = utils.AverageMeter()
            for images, gaze_directions in self.train_loader:
                images, gaze_directions = images.to(self.device), gaze_directions.to(
                    self.device
                )
                self.optimizer.zero_grad()
                outputs = self.model(images)
                train_angular_error.update(
                    utils.compute_angular_error(outputs, gaze_directions).item(),
                    images.size(0),
                )
                loss = self.criterion(outputs, gaze_directions)
                loss.backward()

                unclipped_grads = [p.grad.norm().item() for p in self.model.parameters()]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                clipped_grads = [p.grad.norm().item() for p in self.model.parameters()]

                self.optimizer.step()
                train_loss += loss.item()
                print(
                    f"Train: {i}/{len(self.train_loader.dataset)}, Batch Angular Error: {train_angular_error.val}",
                    end="\r",
                )
                i += images.size(0)
                wandb.log(
                    {
                        "Batch Train Loss": loss.item(),
                        "Batch Angular Error": train_angular_error.val,
                        "Unclipped Grads": sum(unclipped_grads),    
                        "Clipped Grads": sum(clipped_grads),
                        "Train/Pitch": outputs[:, 0].mean().item(),
                        "Train/Yaw": outputs[:, 1].mean().item()
                    }
                )

            # self.scheduler.step()

            self.train_losses.append(train_loss / total_train_batches)

            print()

            val_loss, val_angular_error = self.validate()

            print(
                f"Epoch: {epoch} / {self.num_epochs}, Train Loss: {train_loss / total_train_batches}, Val Loss: {val_loss}, \
                    Train Angular Error: {train_angular_error.avg}, Val Angular Error: {val_angular_error}"
            )

            wandb.log(
            {
                "Train Loss": train_loss / total_train_batches,
                "Train Angular Error": train_angular_error.avg,
                "Val Loss": val_loss ,
                "Val Angular Error": val_angular_error,
            }
        )

    def validate(self, test=False):
        self.model.eval()
        val_angular_error = utils.AverageMeter()
        val_loss = 0.0
        loader = (
            self.test_loader if (test or (self.val_loader is None)) else self.val_loader
        )
        len_loader = len(loader)
        i = 0
        val_angular_error.reset()
        with torch.no_grad():
            for images, gaze_directions in loader:
                images, gaze_directions = images.to(self.device), gaze_directions.to(
                    self.device
                )
                outputs = self.model(images)
                loss = self.criterion(outputs, gaze_directions)
                val_loss += loss.item()
                val_angular_error.update(
                    utils.compute_angular_error(outputs, gaze_directions).item(),
                    images.size(0),
                )
                print(
                    f"Val: {i}/{len(loader.dataset)}, Batch Angular Error: {val_angular_error.val}",
                    end="\r",
                )
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
                # "scheduler": self.scheduler.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "test_participant": self.test_participant,
            }
            torch.save(self.state, self.model_path)

            self.plot_losses()

            if self.val_loader:
                test_loss, test_angular_error = self.validate(test=True)
                print(
                    f"Test Loss: {test_loss}, Test Angular Error: {test_angular_error}"
                )

            wandb.finish()

if __name__ == "__main__":
    vm = VisionMate()
    vm.run()
