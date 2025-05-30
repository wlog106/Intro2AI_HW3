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

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y, self.max_depth)
        self.progress.close()

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        if depth == 0:
            predict = np.argmax(np.bincount(y))
            return {"isLeaf": True, "predict": predict}

        if len(np.unique(y)) == 1:
            return {"isLeaf": True, "predict": y[0]}

        tree = {}
        best_feature_index, best_threshold_index = self._best_split(X, y)
        if best_threshold_index == -1:
            predict = np.argmax(np.bincount(y))
            return {"isLeaf": True, "predict": predict}
        else:
            best_sorted_indices = np.argsort(X[:, best_feature_index])
            X_sorted = X[best_sorted_indices]
            y_sorted = y[best_sorted_indices]
            left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y = self._split_data(
                X_sorted, y_sorted, best_threshold_index
            )
            tree["best_feature_index"] = best_feature_index
            tree["best_threshold"] = (
                            X[best_sorted_indices[best_threshold_index], best_feature_index] 
                            + X[best_sorted_indices[best_threshold_index+1], best_feature_index]
                        )/2
            # left branch
            if left_dataset_y.size == 0:
                predict = np.argmax(np.bincount(y))
                tree["left"] = {"isLeaf": True, "predict": predict}
            elif left_dataset_y.size < 5:
                predict = np.argmax(np.bincount(left_dataset_y))
                tree["left"] = {"isLeaf": True, "predict": predict}
            else:
                tree["left"] = self._build_tree(left_dataset_X, left_dataset_y, depth-1)
            # right branch
            if right_dataset_y.size == 0:
                predict = np.argmax(np.bincount(y))
                tree["right"] = {"isLeaf": True, "predict": predict}
            elif right_dataset_y.size < 5:
                predict = np.argmax(np.bincount(right_dataset_y))
                tree["right"] = {"isLeaf": True, "predict": predict}
            else:
                tree["right"] = self._build_tree(right_dataset_X, right_dataset_y, depth-1)
            tree["isLeaf"] = False
        return tree


    def predict(self, X: np.ndarray)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        predictions = []
        for x in X:
            predict = self._predict_tree(x, self.tree)
            predictions.append(predict)
        return torch.tensor(predictions)

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        if(tree_node["isLeaf"]):
            return tree_node["predict"]

        if(x[tree_node["best_feature_index"]] <= tree_node["best_threshold"]):
            predict = self._predict_tree(x, tree_node["left"])
        else:
            predict = self._predict_tree(x, tree_node["right"])
        
        return predict

    def _split_data(self, X_sorted: np.ndarray, y_sorted: np.ndarray, pos: int):
        # (TODO) split one node into left and right node 

        left_dataset_X = X_sorted[:pos, :]
        left_dataset_y = y_sorted[:pos]
        right_dataset_X = X_sorted[pos: , :]
        right_dataset_y = y_sorted[pos:]
        
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_feature_index = -1
        best_threshold_index = -1
        max_information_gain = -1

        for feature_index in tqdm(range(X.shape[1])):

            sorted_indices = np.argsort(X[:, feature_index])
            X_sorted = X[sorted_indices]
            y_sorted = y[sorted_indices]

            for threshold_index in range(X.shape[0]-1):
                _, left_dataset_y, _, right_dataset_y = self._split_data(X_sorted, y_sorted, threshold_index+1)
                information_gain = (
                    self._entropy(y) 
                    - (left_dataset_y.size/y.size)*self._entropy(left_dataset_y) 
                    - (right_dataset_y.size/y.size)*self._entropy(right_dataset_y)
                )
                if information_gain > max_information_gain:
                    max_information_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold_index = threshold_index

        return best_feature_index, best_threshold_index

    def _entropy(self, y: np.ndarray)->float:
        # (TODO) Return the entropy

        negative_entropy = 0
        for label in np.unique(y):
            if y.size !=0:
                ratio = np.sum(y == label)/y.size
                if ratio > 0:
                    negative_entropy += ratio*np.log2(ratio)
        return -negative_entropy

@torch.no_grad()
def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[np.ndarray, np.ndarray]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    features = []
    labels = []
    pool = nn.AvgPool2d(7, stride=1)
    for images, label in dataloader:
        images = images.to(device)
        feature = pool(model.model.forward_features(images)).squeeze(-1).squeeze(-1)
        labels.append(label.cpu().numpy())
        features.append(feature.cpu().numpy())
    labels = np.concatenate(labels, axis=0)
    features = np.concatenate(features, axis=0)
    return features, labels

@torch.no_grad()
def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    features = []
    paths = []
    pool = nn.AvgPool2d(7, stride=1)
    for images, base_name in dataloader:
        images = images.to(device)
        feature = pool(model.model.forward_features(images)).squeeze(-1).squeeze(-1)
        features.append(feature.cpu().numpy())
        paths = paths + [id for id in base_name]
    features = np.concatenate(features, axis=0)
    return features, paths