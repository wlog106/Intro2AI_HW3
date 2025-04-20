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
        print(f"y in fit: {type(y)}")
        self.data_size = X.shape[0]
        total_steps = 2 ** self.max_depth
        self.progress = tqdm(total=total_steps, desc="Growing tree", position=0, leave=True)
        self.tree = self._build_tree(X, y, self.max_depth)
        self.progress.close()

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        # (TODO) Grow the decision tree and return it
        print(f"y in build: {type(y)}")
        if depth == 0:
            predict = 0
            max_num =0
            for label in range(5):
                num = np.sum(label == y)
                if num > max_num:
                    max_num = num
                    predict = label
            return {"isLeaf": True, "predict": predict}
        # pure node
        if np.max(y) == np.min(y):
            return {"isLeaf": True, "predict": predict}

        tree = {}
        best_feature_index, best_threshold = self._best_split(X, y)
        left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y = self._split_data(
            X, y, best_feature_index, best_threshold
        )
        tree["best_feature_index"] = best_feature_index
        tree["best_threshlod"] = best_threshold
        tree["left"] = self._build_tree(left_dataset_X, left_dataset_y, depth-1)
        tree["right"] = self._build_tree(right_dataset_X, right_dataset_y, depth-1)
        tree["isLeaf"] = False
        return tree


    def predict(self, X: pd.DataFrame)->np.ndarray:
        # (TODO) Call _predict_tree to traverse the decision tree to return the classes of the testing dataset
        predictions = np.ndarray()
        for x in X:
            predict = self._predict_tree(x, self.tree)
            predictions = np.append(predictions, predict)

    def _predict_tree(self, x, tree_node):
        # (TODO) Recursive function to traverse the decision tree
        if(tree_node["isLeaf"]):
            return tree_node["predict"]

        if(x[tree_node["best_feature_index"]] <= tree_node["best_threshold"]):
            predict = self._predict_tree(x, tree_node["left"])
        else:
            predict = self._predict_tree(x, tree_node["right"])
        
        return predict

    def _split_data(self, X: pd.DataFrame, y: np.ndarray, feature_index: int, threshold: float):
        # (TODO) split one node into left and right node 
        left_dataset_X = []
        left_dataset_y = []
        right_dataset_X = []
        right_dataset_y = []
        for i, img_feature in enumerate(X):
            if img_feature[feature_index] <= threshold:
                left_dataset_X.append(img_feature)
                left_dataset_y.append(y[i])
            else:
                right_dataset_X.append(img_feature)
                right_dataset_y.append(y[i])
        left_dataset_X = np.array(left_dataset_X)
        left_dataset_y = np.array(left_dataset_y)
        right_dataset_X = np.array(right_dataset_X)
        right_dataset_y = np.array(right_dataset_y)
        
        return left_dataset_X, left_dataset_y, right_dataset_X, right_dataset_y

    def _best_split(self, X: pd.DataFrame, y: np.ndarray):
        # (TODO) Use Information Gain to find the best split for a dataset
        best_feature_index = 0
        best_threshold = 0
        max_information_gain = 0
        for feature_index in range(X.shape[1]):
            for i in range(X.shape[0]-1):
                threshold = (X[i][feature_index] + X[i+1][feature_index])/2
                _, left_dataset_y, _, right_dataset_y = self._split_data(X, y, feature_index, threshold)
                information_gain = (
                    self._entropy(y) 
                    - (left_dataset_y.size/y.size)*self._entropy(left_dataset_y) 
                    - (right_dataset_y.size/y.size)*self._entropy(right_dataset_y)
                )
                if information_gain > max_information_gain:
                    max_information_gain = information_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _entropy(self, y: np.ndarray)->float:
        # (TODO) Return the entropy
        negative_entropy = 0
        for label in range(5):
            ratio = np.sum(y == label)/y.size
            negative_entropy += ratio*np.log2(ratio)
        return -negative_entropy

@torch.no_grad()
def get_features_and_labels(model: ConvNet, dataloader: DataLoader, device)->Tuple[np.ndarray, np.ndarray]:
    # (TODO) Use the model to extract features from the dataloader, return the features and labels
    features = []
    labels = []
    for images, label in dataloader:
        images = images.to(device)
        feature = model(images)
        labels.append(label.cpu().numpy())
        features.append(feature.cpu().numpy())
    labels = np.concatenate(labels, axis=0)
    features = np.concatenate(features, axis=0)
    return features, labels

@torch.no_grad()
def get_features_and_paths(model: ConvNet, dataloader: DataLoader, device)->Tuple[List, List]:
    # (TODO) Use the model to extract features from the dataloader, return the features and path of the images
    features = []
    for images, base_name in dataloader:
        images = images.to(device)
        feature = model(images)
        features.append(feature.cpu().numpy())
    features = np.concatenate(features, axis=0)
    paths = ["data/test/" + id + ".jpg" for id in base_name.tolist()]
    return features, paths