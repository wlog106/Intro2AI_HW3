import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from typing import List, Tuple

"""
Notice:
    1) You can't add any additional package
    2) You can add or remove any function "except" fit, _build_tree, predict
    3) You can ignore the suggested data type if you want
"""

class ConvNet(nn.Module): # Don't change this part!
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x
    
class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        root = {}

    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        raise NotImplementedError

    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        left_dataset_X = []
        left_dataset_y = []
        right_dataset_X = []
        right_dataset_y = []
        for img_feature in X:
            if img_feature[feature_index] <= threshold:
                left_dataset_X.append(img_feature)
                left_dataset_y.append(y[feature_index])
            else:
                right_dataset_X.append(img_feature)
                right_dataset_y.append(y[feature_index])
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_feature_index = 0
        max_information_gain = 0
        for feature_index in range(X.shape[1]):
            for i in range(X.shapep[0]):
                left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y = self._split_data(X, y, feature_index, threshold)
                information_gain = (
                    self._entropy(y) 
                    - (len(left_dataset_y)/len(y))*self._entropy(left_dataset_y) 
                    - (len(right_dataset_y)/len(y))*self._entropy(right_dataset_y)
                )
                if information_gain > max_information_gain:
                    max_information_gain = information_gain
                    best_feature_index = feature_index

        return best_feature_index, best_threshold

    def _entropy(y: np.ndarray)->float:
        # (TODO) Return the entropy
        raise NotImplementedError

@torch.no_grad()
def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    labels = []
    for images, label in dataloader:
        images = images.to(device)
        features: torch.Tensor = model.foward_features(images)
        labels = labels + label
    pool = nn.AvgPool2d(7, stride=1)
    features = pool(features).cpu().tolist()
    return features, labels

@torch.no_grad()
def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    for images, base_name in dataloader:
        images = images.to(device)
        features = model.forward_features(images)
    pool = nn.AvgPool2d(7, stride=1)
    features = pool(features).cpu().tolist()
    paths = ["data/test/" + id + ".jpg" for id in base_name.tolist()]
    return features, paths