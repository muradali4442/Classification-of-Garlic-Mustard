from torch.utils.tensorboard import SummaryWriter
import torch
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

#Dataloader
transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

dataset = datasets.ImageFolder(root='/Users/muradali/Desktop/Data_Training_2/Train',
                                    transform = transforms)
testloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              shuffle=True, num_workers=0)

#Run predictions
model = models.resnet18(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('2_stage_model.pt'))
model.eval()

writer = SummaryWriter('eval_runs')
classes = ('Flowers', 'No_Flowers')
class_probs = []
class_preds = []
labs = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = model(images)
        labs.append(labels)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        _, class_preds_batch = torch.max(output, 1)
        class_probs.append(class_probs_batch)
        class_preds.append(class_preds_batch)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_preds = torch.cat(class_preds)
labs = torch.cat(labs)


for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, labs)

#Get model output
preds = test_preds.numpy()
probs = test_probs.numpy()
model_output = np.column_stack((preds, probs))
model_output = pd.DataFrame(model_output, columns = ['Label', 'f_prob', 'nf_prob'])
model_output['true_label'] = list(labs)

#Plot and save data
plt.hist(model_output['f_prob'], bins=100)
model_output.to_csv('eval_output_2.csv')