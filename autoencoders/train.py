import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import random
import os
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import gudhi as gd
import gudhi.representations

from model import Autoencoder


torch.manual_seed(0)


#----------------------------------------
# Datasets
#----------------------------------------
data_dir = 'data'

train_dataset = torchvision.datasets.MNIST(data_dir, transform=transforms.ToTensor(), train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, transform=transforms.ToTensor(), train=False, download=True)

m = len(train_dataset)
batch_size = 256
latent_space_dim = 3
lr = 1e-3
num_epochs = 3

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

#----------------------------------------
# Make sure the model works
#----------------------------------------
model = Autoencoder(latent_space_dim=latent_space_dim, lr=lr)

print('-' * 50)

img, _ = test_dataset[0]
img = img.unsqueeze(0) # Add the batch dimension in the first axis
print('Original image shape:', img.shape)

img_enc = model.encode(img)
print('Encoded image shape:', img_enc.shape)

dec_img = model.decode(img_enc)
print('Decoded image shape:', dec_img.shape)

print('-' * 50)

#----------------------------------------
# Training/testing methods
#----------------------------------------

# hybrid loss that combines standard MSE and MSE between persistence landscapes
def _loss(orig, decoded):
    mse_loss = ((decoded - orig) ** 2).mean()

    #! implement landscapes stuff
    topo_loss = 0

    #! coefficients
    return mse_loss + topo_loss

# template for train/test epochs
# returns average loss from all batches in the epoch
def _epoch(model, dataloader, optimize):
    losses = []

    for img, _ in dataloader:
        enc = model.encode(img)
        dec = model.decode(enc)

        loss = _loss(img, dec)
        with torch.no_grad():
            losses.append(loss.item())

        if optimize:
            model.minimize(loss)
    
    return np.mean(losses)

# training epoch: get data from training dataloader and optimize model
def train_epoch(model):
    return _epoch(model, train_loader, True)

# testing epoch: get data from testing dataloader and don't optimize model
def test_epoch(model):
    return _epoch(model, test_loader, False)

#----------------------------------------
# Persistence helper methods
#----------------------------------------
def get_encoded_points_and_labels(model, image_set, num_points=None):
    points = []
    labels = []
    if num_points is None:
        num_points = len(image_set)
    for i in range(num_points):
        img = image_set[i][0].unsqueeze(0)
        label = image_set[i][1]
        labels.append(label)
        with torch.no_grad():
            encoded_img = model.encode(img)
            rec_img = model.decode(encoded_img)
            encoded_img = encoded_img.flatten().numpy()
        points.append(encoded_img)
    return points, labels

def get_encoded_points_and_labels_by_index(model, image_set, num_points=None, start_index=0):
    points = []
    labels = []
    if num_points is None:
        num_points = len(image_set)
    for i in range(start_index, start_index + num_points):
        img = image_set[i][0].unsqueeze(0)
        label = image_set[i][1]
        labels.append(label)
        with torch.no_grad():
            encoded_img = model.encode(img)
            rec_img = model.decode(encoded_img)
            encoded_img = encoded_img.flatten().numpy()
        points.append(encoded_img)
    return points, labels

def make_simplicial_complex(points, diameter):
    skeleton = gd.RipsComplex(points=points, max_edge_length=diameter)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=2)
    return simplex_tree

def get_persistence_features(simplex_tree):
    simplex_tree.persistence()
    # barcodes = simplex_tree.persistence()
    zero_dim_features = simplex_tree.persistence_intervals_in_dimension(0)
    one_dim_features = simplex_tree.persistence_intervals_in_dimension(1)
    two_dim_features = simplex_tree.persistence_intervals_in_dimension(2)

    features = [zero_dim_features, one_dim_features, two_dim_features]
    # features = normalize_features(features, 1000)
    return features

def normalize_features(features, max_value):
    for k_dim_features in features:
        for feature in k_dim_features:
            if feature[1] == float('inf'):
                feature[1] = max_value
    return features

def get_persistence_landscapes(features, num_landscapes, diameter, resolution=100):
    landscape = gd.representations.Landscape(num_landscapes=num_landscapes, 
                                             resolution=resolution,
                                             sample_range=[0, diameter])
    landscape_vectors = np.zeros((len(features), num_landscapes * resolution))
    for i, k_dim_features in enumerate(features):
        if len(k_dim_features) > 0:
            landscape_vector = landscape.fit_transform([k_dim_features])
            landscape_vectors[i] += landscape_vector.flatten()
    return np.array(landscape_vectors)

def get_all_encoded_points(model, image_set):
    encoded_images = []
    labels = []
    for sample in tqdm(image_set):
        img = sample[0].unsqueeze(0)
        label = sample[1]
        # Encode image
        with torch.no_grad():
            encoded_img = model.encode(img)
        # Append to list
        encoded_img = encoded_img.flatten().numpy()
        encoded_images.append(encoded_img)
        labels.append(label)
    return np.array(encoded_images), np.array(labels)

def set_diameter(image_set, num_points):
    '''Sample num_points random points from the encoded points of the image set, find the diameter of the point cloud, 
    return the diameter / 2'''
    points = get_all_encoded_points(model, image_set)[0]
    sampled_point_indices = np.random.choice(len(points), size=num_points, replace=False)
    sampled_points = points[sampled_point_indices, :]
    diameter = calculate_point_cloud_diameter(sampled_points)
    return diameter / 2

def calculate_point_cloud_diameter(points):
    max_diameter = 0
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            max_diameter = max(max_diameter, distance)
    return max_diameter

def get_test_class_samples(classes):
    image_set = [sample for sample in test_dataset if sample[1] in classes]
    return image_set

# ******************************
def plot_average_persistence_landscapes(model, image_set, num_landscapes=10, num_points=100):
    num_batches = len(image_set) // num_points
    landscape_vectors_sum = np.zeros((3, num_landscapes * num_points))
    diameter = set_diameter(image_set, num_points)
    for batch_num in range(num_batches):
        start_index = batch_num * num_points
        points = get_encoded_points_and_labels_by_index(model, image_set, num_points, start_index)[0]
        simplex_tree = make_simplicial_complex(points, diameter)
        features = get_persistence_features(simplex_tree)
        landscape_vectors = get_persistence_landscapes(features, num_landscapes, diameter)
        landscape_vectors_sum += landscape_vectors

    landscape_vectors_average = landscape_vectors_sum / num_points
    for i, landscape_vector in enumerate(landscape_vectors_average):
        plt.clf()
        if len(landscape_vector) > 0:
            plt.title(f'{i}-dim')
            for i in range(num_landscapes):
                plt.plot(landscape_vector[i * num_points:(i + 1) * num_points])
        plt.show()
# ******************************

#----------------------------------------
# Training loop
#----------------------------------------
def train(model, epochs):
    print('Starting training')
    train_losses = []
    test_losses = []

    for _ in tqdm(range(epochs)):
        train_losses.append(train_epoch(model))
        test_losses.append(test_epoch(model))

    print('-' * 50)
    return train_losses, test_losses

train_losses, test_losses = train(model, num_epochs)
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.show()
