from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import os
import time
import numpy as np

test_path = 'garbage/garbage classification/Garbage classification'
test_txt_path = 'garbage/one-indexed-files-notrash_test.txt'
model_path = 'app/model.pth'

class MyDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None):
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        self.class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
        self.img_list = [os.path.join(img_dir, ''.join(filter(str.isalpha, line.split('.')[0])), line.split()[0]) for line in lines]
        self.label_list = [self.class_names[int(line.split()[1]) - 1] for line in lines]  # convert indices to class names
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.label_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

def class_to_index(class_name):
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    return class_names.index(class_name)

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = MyDataset(test_txt_path, test_path, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = torchvision.models.resnet50()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
  
model = model.to(device)

model.load_state_dict(torch.load(model_path))
model.eval()

top1_correct = 0
top5_correct = 0
total = 0
class_correct = [0] * 6
class_total = [0] * 6
batch_times = []

with torch.no_grad():
  predictions = []
  true_labels = []
  for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = torch.tensor([class_to_index(label) for label in labels]).to(device)

    start_time = time.time()
    outputs = model(inputs)
    probabilities = torch.softmax(outputs, dim=1)
    end_time = time.time()
    batch_times.append(end_time - start_time)

    _, predicted = torch.max(probabilities, 1)
    total += labels.size(0)
    top1_correct += (predicted == labels).sum().item()

    predictions.extend(predicted.tolist())
    true_labels.extend(labels.tolist())

    _, top5_predicted = torch.topk(probabilities, k=5, dim=1)
    for i in range(labels.size(0)):
      if labels[i] in top5_predicted[i]:
        top5_correct += 1

    for i in range(labels.size(0)):
      label = labels[i]
      class_correct[label] += (predicted[i] == label).item()
      class_total[label] += 1

predictions = torch.tensor(predictions).cpu().numpy()
true_labels = torch.tensor(true_labels).cpu().numpy()

top1_error = 100 * (1 - top1_correct / total)
top5_error = 100 * (1 - top5_correct / total)

print('Top-1 Error: %.2f %%' % top1_error)
print('Top-5 Error: %.2f %%' % top5_error)
print('')

class_accuracy_top1 = [100 * class_correct[i] / class_total[i] for i in range(6)]
class_accuracy_top5 = [100 * top5_correct / total for _ in range(6)]

for i in range(6):
    class_name = test_dataset.class_names[i]
    accuracy_top1 = class_accuracy_top1[i]
    accuracy_top5 = class_accuracy_top5[i]
    print('Accuracy of %s - Top 1: %.2f %%' % (class_name, accuracy_top1))
    print('Accuracy of %s - Top 5: %.2f %%' % (class_name, accuracy_top5))
    print('')

avg_batch_time = sum(batch_times) / len(batch_times)
print('Average Batch Inference Time: %.4f seconds' % avg_batch_time)
print('')

ap_scores = []
for i in range(6):
    class_predictions = (predictions == i)
    class_true_labels = (true_labels == i)
    
    tp = np.sum(class_predictions & class_true_labels)
    fp = np.sum(class_predictions & (~class_true_labels))
    
    precision = tp / (tp + fp)
    recall = tp / np.sum(class_true_labels)
    
    ap = precision * recall
    
    ap_scores.append(ap)

mAP = sum(ap_scores) / len(ap_scores)

for i in range(6):
    class_name = test_dataset.class_names[i]
    ap = ap_scores[i]
    print('AP of %s: %.4f' % (class_name, ap))
    
print('')
print('mAP: %.4f' % mAP)