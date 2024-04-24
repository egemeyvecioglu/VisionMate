import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import matplotlib.pyplot as plt
from dataloader import MPIIFaceGazeDataset, Gaze360Dataset, MPIIFaceGazeProcessedDataset
import wandb
from model import FastViTMLP, FastViTL2CS, FasterViT
from torch.utils.data import random_split
import os
import utils

class VisionMate:
    def __init__(self) -> None:
        self.num_epochs = 75
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.model_path = "mpii.pth"
        self.method = "FasterViT"
        self.dataset = ""
        self.data_dir = (
                 "/home/kovan-beta/Desktop/visionmate/datasets/MPIIFaceGaze"              if self.dataset == "MPIIFaceGaze" 
            else "/home/kovan-beta/Desktop/visionmate/datasets/MPIIFaceGazeProcessed"     if self.dataset == "MPIIFaceGazeProcessed"
            else "/home/kovan-beta/Desktop/visionmate/datasets/Gaze360/"
        )
        self.train_losses = []
        self.val_losses = []
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available else "cpu"
        )
        self.current_epoch = 0

        self.clip_grads = True

        self.state = None
        torch.set_float32_matmul_precision('high')
        if self.method == "FastViT":
            self.model = FastViTMLP(self.device)
            self.criterion = nn.MSELoss()
        elif self.method == "FasterViT":
            self.model = FasterViT(self.device)
            self.criterion = nn.MSELoss()
        elif self.method == "L2CS":
            self.model = FastViTL2CS(self.device, 28)
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.reg_criterion = nn.MSELoss().to(self.device)
            self.softmax = nn.Softmax(dim=1).to(self.device)
            self.idx_tensor = torch.FloatTensor([idx for idx in range(28)]).to(self.device)
            self.alpha = 1


        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # self.optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4
        # )

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1, eta_min=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-7)  # cosine annealing learning rate

        self.test_participant = None

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
            self.test_participant = self.state["test_participant"]
            self.num_epochs = self.num_epochs - epoch

        self.train_loader, self.val_loader, self.test_loader = self.load_dataset()

        self.log_wandb = True
        if self.log_wandb:
            wandb.init(
                project="VisionMate",
                config={
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "num_epochs": self.num_epochs,
                },
            )  # Replace with your details

    def load_dataset(self):

        size = 224 if self.method == "FasterViT" else 256

        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=size,scale=(0.8,1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        val_transform = transforms.Compose([
            transforms.Resize((size,size)),
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

        elif self.dataset == "MPIIFaceGazeProcessed":
            
            # randomly leave one participant out for testing
            randint = torch.randint(0, 15, (1,))
            self.test_participant = f"p{randint.item():02d}"

            print("Test participant: ", self.test_participant)

            train_dataset = MPIIFaceGazeProcessedDataset(
                self.data_dir, self.test_participant, train=True, transform=train_transform
            )
            test_dataset = MPIIFaceGazeProcessedDataset(
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
    
    def validate_l2cs(self, test=False):
        self.model.eval()
        val_pitch_error = utils.AverageMeter()
        val_yaw_error = utils.AverageMeter()
        val_pitch_loss = 0.0
        val_yaw_loss = 0.0
        loader = (
            self.test_loader if (test or (self.val_loader is None)) else self.val_loader
        )
        len_loader = len(loader)
        i = 0
        val_pitch_error.reset()
        val_yaw_error.reset()
        with torch.no_grad():
            for images, labels, cont_labels in loader:
                images, labels, cont_labels = images.to(self.device), labels.to(self.device), cont_labels.to(self.device)
                
                label_pitch = labels[:, 0]
                label_yaw = labels[:, 1]

                label_pitch_cont = cont_labels[:, 0]
                label_yaw_cont = cont_labels[:, 1]

                pre_yaw_gaze, pre_pitch_gaze = self.model(images)

                # cross entropy loss
                loss_pitch = self.criterion(pre_pitch_gaze, label_pitch)
                loss_yaw = self.criterion(pre_yaw_gaze, label_yaw)

                # MSE loss
                pitch_predicted = self.softmax(pre_pitch_gaze)
                yaw_predicted = self.softmax(pre_yaw_gaze)

                pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, 1) * 3 - 42
                yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, 1) * 3 - 42

                loss_pitch_reg = self.reg_criterion(pitch_predicted, label_pitch_cont)
                loss_yaw_reg = self.reg_criterion(yaw_predicted, label_yaw_cont)

                loss_pitch = loss_pitch + self.alpha * loss_pitch_reg
                loss_yaw = loss_yaw + self.alpha * loss_yaw_reg

                val_pitch_loss += loss_pitch.item()
                val_yaw_loss += loss_yaw.item()

                i += images.size(0)

        print()

        if not test:
            self.val_losses.append((val_pitch_loss / len_loader, val_yaw_loss / len_loader))

        return val_pitch_loss / len_loader, val_yaw_loss / len_loader
    
    def train_l2cs(self):
        print("Training starts. Device: ", self.device)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch} learning rate: {self.scheduler.get_last_lr()[-1]}")
            self.model.train()
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = 0.0
            total_train_batches = len(self.train_loader)
            i = 0

            pitch_loss_meter = utils.EMAMeter()
            yaw_loss_meter = utils.EMAMeter()
            angle_error_meter = utils.EMAMeter()

            for images, labels, cont_labels in self.train_loader:
                images, labels, cont_labels = images.to(self.device), labels.to(self.device), cont_labels.to(self.device)
                
                label_pitch = labels[:, 0]
                label_yaw = labels[:, 1]

                label_pitch_cont = cont_labels[:, 0]
                label_yaw_cont = cont_labels[:, 1]

                self.optimizer.zero_grad()
                pre_yaw_gaze, pre_pitch_gaze = self.model(images)

                # cross entropy loss
                loss_pitch = self.criterion(pre_pitch_gaze, label_pitch)
                loss_yaw = self.criterion(pre_yaw_gaze, label_yaw)

                # MSE loss
                pitch_predicted = self.softmax(pre_pitch_gaze)
                yaw_predicted = self.softmax(pre_yaw_gaze)

                pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, 1) * 3 - 42
                yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, 1) * 3 - 42

                # concatenate the predicted values. size = (batch_size, 2)
                pred = torch.cat((pitch_predicted.unsqueeze(1), yaw_predicted.unsqueeze(1)), 1)
                
                angle_error = utils.compute_angular_error(pred, cont_labels)
                angle_error_meter.update(angle_error)
                

                loss_pitch_reg = self.reg_criterion(pitch_predicted, label_pitch_cont)
                loss_yaw_reg = self.reg_criterion(yaw_predicted, label_yaw_cont)

                loss_pitch = loss_pitch + self.alpha * loss_pitch_reg
                loss_yaw = loss_yaw + self.alpha * loss_yaw_reg

                pitch_loss_meter.update(loss_pitch.item())
                yaw_loss_meter.update(loss_yaw.item())

                sum_loss_pitch_gaze += loss_pitch
                sum_loss_yaw_gaze += loss_yaw

                loss_seq = [loss_pitch, loss_yaw]
                grad_seq = [torch.tensor(1.0).to(self.device) for _ in loss_seq]
                self.optimizer.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                self.optimizer.step()
                
                print(
                    f"Train: {i}/{len(self.train_loader.dataset)}, Pitch Loss: {pitch_loss_meter.avg}, Yaw Loss: {yaw_loss_meter.avg}, Angular Error: {angle_error_meter.avg}",
                    end="\r",
                )

                i += images.size(0)
                self.log(
                    {
                        "Train Pitch Loss": loss_pitch.item(),
                        "Train Yaw Loss": loss_yaw.item(),
                        "Angular Error": angle_error,
                    }
                )

            self.train_losses.append(
                (sum_loss_pitch_gaze / total_train_batches, sum_loss_yaw_gaze / total_train_batches)
            )

            print()

            val_pitch_loss, val_yaw_loss, val_angular_error = self.validate_l2cs()

            print(
                f"Epoch: {epoch} / {self.num_epochs}, Train Pitch Loss: {sum_loss_pitch_gaze / total_train_batches}, Train Yaw Loss: {sum_loss_yaw_gaze / total_train_batches}, \
                    Val Pitch Loss: {val_pitch_loss}, Val Yaw Loss: {val_yaw_loss}"
            )

            self.log(
                {
                    "Train Pitch Loss": sum_loss_pitch_gaze / total_train_batches,
                    "Train Yaw Loss": sum_loss_yaw_gaze / total_train_batches,
                    "Val Pitch Loss": val_pitch_loss,
                    "Val Yaw Loss": val_yaw_loss,
                }

            )

    def train(self):
        print("Training starts. Device: ", self.device)

        best_val_angular_error = 180

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch} learning rate: {self.scheduler.get_last_lr()[-1]}")
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
                if self.clip_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                clipped_grads = [p.grad.norm().item() for p in self.model.parameters()]

                self.optimizer.step()
                train_loss += loss.item()
                print(
                    f"Train: {i}/{len(self.train_loader.dataset)}, Batch Angular Error: {train_angular_error.val}",
                    end="\r",
                )

                i += images.size(0)
                self.log(
                    {
                        "Batch Train Loss": loss.item(),
                        "Batch Angular Error": train_angular_error.val,
                        "Unclipped Grads": sum(unclipped_grads),    
                        "Clipped Grads": sum(clipped_grads),
                        "Train/Pitch": outputs[:, 0].mean().item(),
                        "Train/Yaw": outputs[:, 1].mean().item()
                    }
                )


            self.train_losses.append(train_loss / total_train_batches)

            print()

            val_loss, val_angular_error = self.validate()

            if val_angular_error < best_val_angular_error:
                best_val_angular_error = val_angular_error
                self.save()

            self.scheduler.step()
            
            print(
                f"Epoch: {epoch+1} / {self.num_epochs}, Train Loss: {train_loss / total_train_batches:.4f}, Val Loss: {val_loss:.4f}, \
                    Train Angular Error: {train_angular_error.avg:.4f}, Val Angular Error: {val_angular_error:.4f}"
            )

            self.log(
            {
                "Train Loss": train_loss / total_train_batches,
                "Train Angular Error": train_angular_error.avg,
                "Val Loss": val_loss ,
                "Val Angular Error": val_angular_error,
                "Train/learning rate": self.scheduler.get_last_lr()[-1]
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

    def log(self, dict):
        if self.log_wandb:
            wandb.log(dict)
    
    def save(self):
        self.state = {
            "epoch": self.current_epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "test_participant": self.test_participant,
        }
        torch.save(self.state, self.model_path)

    def run(self):
        try:
            if self.method == "L2CS":
                self.train_l2cs()
            else:
                self.train()
        
        except KeyboardInterrupt or Exception as e:
            print("Error: ", e)

        finally:

            self.plot_losses()

            if self.val_loader:
                test_loss, test_angular_error = self.validate(test=True)
                print(
                    f"Test Loss: {test_loss}, Test Angular Error: {test_angular_error}"
                )

            if self.log_wandb:
                wandb.finish()

if __name__ == "__main__":
    vm = VisionMate()
    vm.run()
