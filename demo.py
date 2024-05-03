"""Loads a trained model and runs a demo on each video frame read from a webcam."""
import os
import cv2
import torch
import numpy as np
from model import FastViTMLP,FasterViT, FastViTL2CS, ResNET50L2CS, FastViTL2CS
import utils
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import time
import torchvision

class GazeEstimator:
    def __init__(self, model_name, model_type, device):
        self.device = device
        self.model_name = model_name
        self.model_type = model_type

        if self.model_type == "FastViTMLP":
            self.model = FastViTMLP(self.device)
        elif self.model_type == "FasterViT":
            self.model = FasterViT(self.device)
        elif self.model_type == "FastViTL2CS":
            self.model = FastViTL2CS(self.device, 28)
        elif self.model_type == "ResNET50L2CS":
            self.model = ResNET50L2CS(self.device, 90, torchvision.models.resnet.Bottleneck, [3, 4, 6, 3])
            self.softmax = torch.nn.Softmax(dim=1)
            self.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(self.device)
        else:
            raise ValueError(f"Model type {self.model_type} not supported.")

        state = torch.load(self.model_name, map_location=self.device)
        if self.model_type == "ResNET50L2CS":
            self.model.load_state_dict(state)
        else:
            self.model.load_state_dict(state["state_dict"])
        self.model.eval()

        size = 448 if self.model_type == "ResNET50L2CS" else  "256" if self.model_type == "FastViTMLP" else 224
        self.val_transform = transforms.Compose([
                    transforms.Resize((size, size)),       
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    def estimate_gaze(self, image):
        image = self.val_transform(Image.fromarray(image)).to(self.device)

        with torch.no_grad():
            image = image.unsqueeze(0)
 
            if self.model_type == "ResNET50L2CS":
                # Predict 
                gaze = utils.L2CS_final_output(self.model, image)
                return gaze

            gaze = self.model(image).cpu().detach().numpy()
            return gaze


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.cuda.is_available() else "cpu")
    model_type = "ResNET50L2CS"
    model_name = "/home/kovan-beta/L2CS-Net/models/L2CSNet_gaze360.pkl"
    gaze_estimator1 = GazeEstimator(model_name, model_type, device)

    model_type = "FasterViT"
    model_name = "/home/kovan-beta/Desktop/visionmate/VisionMate/models/fastervit_gaze360-2.pth"
    gaze_estimator2 = GazeEstimator(model_name, model_type, device)

    face_detector = YOLO("/home/kovan-beta/Desktop/visionmate/VisionMate/models/face_detection/yolov8m-face.pt")
    crop_head = True

    cap = cv2.VideoCapture(0)

    while True:
        start = time.time()

        ret, frame = cap.read()
        
        if crop_head:
            results = face_detector(frame, stream=True, conf=0.5, verbose=False)

            if results is not None:
                for result in results:
                    boxes = result.boxes

                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        offset = 20  # Adjust this value as needed
                        x1 = max(0, x1 - offset)
                        y1 = max(0, y1 - offset)
                        x2 = min(frame.shape[1], x2 + offset)
                        y2 = min(frame.shape[0], y2 + offset)


                        frame = frame[y1:y2, x1:x2]

        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        image_center_x = (x1 + x2) // 2
        image_center_y = (y1 + y2) // 2


        gaze = gaze_estimator1.estimate_gaze(frame)
        pitch, yaw = gaze[0][0], gaze[1][0]
        print("L2CS: ", pitch, yaw)
        utils.draw_gaze(x1, y1, x2, y2, frame, (pitch, yaw), color=(0, 255, 0)) # GREEN


        gaze2 = gaze_estimator2.estimate_gaze(frame)
        pitch2, yaw2 = gaze2[0, 0], gaze2[0, 1]
        print("FasterViT: ", pitch2, yaw2)
        utils.draw_gaze(x1, y1, x2, y2, frame, (pitch2, yaw2), color=(0, 0, 255)) # RED


        # print(gaze)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        print("FPS: ", 1 / (time.time() - start), end="\r")

    cap.release()
    cv2.destroyAllWindows()