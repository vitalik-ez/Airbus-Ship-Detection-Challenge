import os
import time
import glob
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp



class ShipSegmentation:

    def __init__(self, model_path, device, encoder='mit_b0', encoder_weights='imagenet'):
        self.device = device
        self._model = self._load_model(model_path)
        self._encoder = encoder
        self._encoder_weights = encoder_weights
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self._encoder, self._encoder_weights)
        self.model_preprocessing = self._get_preprocessing(preprocessing_fn)

    def _load_model(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.to(self.device)
        model.eval()
        return model


    def _to_tensor(self, x):
        return x.transpose(2, 0, 1).astype('float32')

    def _get_preprocessing(self, preprocessing_fn):
        _transform = [
            transforms.Lambda(lambda x: preprocessing_fn(x)),
            transforms.Lambda(lambda x: self._to_tensor(x)),
        ]
        return transforms.Compose(_transform)
    
    def preprocessing(self, image):
        dim = (544, 544)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        image = self.model_preprocessing(image)
        return torch.from_numpy(image).to(self.device).unsqueeze(0)
    
    def predict(self, orig_image):
        image = self.preprocessing(orig_image)
        ship_mask = self._model.predict(image)
        ship_mask = (ship_mask.squeeze().cpu().numpy().round() * 255).astype(np.uint8)
        ship_mask = cv2.resize(ship_mask, orig_image.shape[:2][::-1], interpolation = cv2.INTER_AREA)
        return ship_mask



def main():

    weights_dir = "weights"
    model_path = os.path.join(weights_dir, "mit_b0_18_epoch.pth")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ship = ShipSegmentation(model_path=model_path, device=device)

    images = glob.glob("./Airbus_Ship_Detection_splited_dataset/test/*")
    image_path = random.choice(images)

    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    predicted_mask = ship.predict(rgb_image)
    predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)

    cv2.imwrite("./output/3.jpg", np.hstack((bgr_image, predicted_mask)))



if __name__ == "__main__":
    main()