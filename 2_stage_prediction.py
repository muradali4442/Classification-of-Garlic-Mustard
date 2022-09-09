from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import os
class CustomDataSet(Dataset):
    def __init__(self,files, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        for i in self.files:
            img_path=os.path.join(self.root_dir, i)
            image = Image.open(img_path).convert("RGB")
            tensor_image = self.transform(image)
            return tensor_image

#def locations
input_file = 'Train.CSV'
img_dir = '/Users/muradali/Desktop/Data_Training_2/predictions'

#Dataloader
transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

files=os.listdir(img_dir)
dataset = CustomDataSet(files, root_dir=img_dir,
                                 transform = transforms)
testloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

#Run predictions
model = models.resnet18(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('2_stage_eval_putput.pt'))
model.eval()

classes = ('Flowers', 'No_Flowers')
class_probs = []
class_preds = []
with torch.no_grad():
    for data in testloader:
        images = data
        output = model(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)

        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)

#Get model output
preds = test_preds.numpy()
probs = test_probs.numpy()
model_output = np.column_stack((preds, probs))
model_output = pd.DataFrame(model_output, columns = ['Label', 'f_prob', 'nf_prob'])
model_output.to_csv('2_stage_model_predictions.csv')
