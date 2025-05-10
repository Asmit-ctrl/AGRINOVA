import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

import nltk
nltk.download('punkt_tab')

directory_path = 'D:\\projectllp\\logathon\\college\\NewIntents'

# Load JSON files from directory
json_files = []
for filename in os.listdir(directory_path):
    if 'dataset' in filename and filename.endswith('.json'):
        json_files.append(os.path.join(directory_path, filename))

num_epochs = 3000
batch_size = 16
learning_rate = 0.0001
hidden_size = 256

all_words = []
tags = []
xy = []

# Extract patterns and tags from JSON files
for json_file in json_files:
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            intents = json.load(f)
            
        for intent in intents['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                w = tokenize(pattern)
                all_words.extend(w)
                xy.append((w, tag))
    else:
        print(f"Warning: {json_file} does not exist. Skipping this file.")

ignoreLetters = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignoreLetters]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X = []
y = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)

input_size = len(X[0])
output_size = len(tags)

class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Cross-Validation Parameters
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_train_accuracies = []
fold_test_accuracies = []
fold_losses = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"Processing Fold {fold + 1}/{k_folds}...")
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    train_dataset_fold = ChatDataset(X_train_fold, y_train_fold)
    test_dataset_fold = ChatDataset(X_test_fold, y_test_fold)
    
    train_loader_fold = DataLoader(dataset=train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader_fold = DataLoader(dataset=test_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Reset model, optimizer, and loss for each fold
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    fold_loss = []
    for epoch in range(num_epochs):
        model.train()
        for (words, labels) in train_loader_fold:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        fold_loss.append(loss.item())
    
    # Training Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for (words, labels) in train_loader_fold:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    fold_train_accuracy = correct / total
    fold_train_accuracies.append(fold_train_accuracy)
    
    # Test Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for (words, labels) in test_loader_fold:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    fold_test_accuracy = correct / total
    fold_test_accuracies.append(fold_test_accuracy)

    fold_losses.append(fold_loss[-1])

# Plot Cross-Validation Results
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.bar(range(1, k_folds + 1), fold_train_accuracies, color='blue', label='Train Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Across Folds')
plt.legend()

plt.subplot(2, 2, 2)
plt.bar(range(1, k_folds + 1), fold_test_accuracies, color='orange', label='Test Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Across Folds')
plt.legend()

plt.subplot(2, 2, 3)
plt.bar(range(1, k_folds + 1), fold_losses, color='red', label='Loss')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Loss Across Folds')
plt.legend()

# Updated Cross-Validation Results Plotting
plt.figure(figsize=(14, 10))

# Training Accuracy
plt.subplot(2, 2, 1)
plt.bar(range(1, k_folds + 1), fold_train_accuracies, color='blue', label='Train Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Across Folds')
plt.legend()

# Testing Accuracy
plt.subplot(2, 2, 2)
plt.bar(range(1, k_folds + 1), fold_test_accuracies, color='orange', label='Test Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy Across Folds')
plt.legend()

# Training Loss
plt.subplot(2, 2, 3)
plt.bar(range(1, k_folds + 1), fold_losses, color='red', label='Training Loss')
plt.xlabel('Fold')
plt.ylabel('Loss')
plt.title('Training Loss Across Folds')
plt.legend()

# Combined Accuracy and Loss
plt.subplot(2, 2, 4)
plt.plot(range(1, k_folds + 1), fold_train_accuracies, label='Train Accuracy', marker='o', color='blue')
plt.plot(range(1, k_folds + 1), fold_test_accuracies, label='Test Accuracy', marker='x', color='orange')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy Across Folds')
plt.legend()

plt.tight_layout()
plt.show()

if __name__ == '__main__':
    pass