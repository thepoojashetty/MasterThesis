import os
import numpy as np
from skimage import io
import torch

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = io.imread(os.path.join(folder, filename))
            if img is not None:
                images[filename] = img
    return images

def dice_coefficient(y_true, y_pred,classes_to_include, eps=1e-7):
    y_true = torch.from_numpy(y_true.astype(np.int32)).flatten()
    y_pred = torch.from_numpy(y_pred.astype(np.int32)).flatten()

    dice_score = []
    for cls in classes_to_include:
        y_true_cls = (y_true == cls).to(torch.float32)
        y_pred_cls = (y_pred == cls).to(torch.float32)

        intersection = torch.sum(y_true_cls * y_pred_cls)

        union = torch.sum(y_true_cls) + torch.sum(y_pred_cls)

        dice = (2. * intersection + eps) / (union + eps)
        dice_score.append(dice.item())
    return np.mean(dice_score)

# class DiceLoss(torch.nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#         self.eps=1e-7

#     def forward(self, y_true, y_pred):
#         return 1- dice_coefficient(y_true, y_pred,self.eps)
