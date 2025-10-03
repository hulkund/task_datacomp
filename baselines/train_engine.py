import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from train_on_subset_classification import train_classification, evaluate_classification
from train_on_subset_regression import train_regression, evaluate_regression
from train_on_subset_reid import train_reid, evaluate_reid

class TrainEngine():
    def __init__(self, training_task, **kwargs):
        self.training_task = training_task
        self.kwargs = kwargs
        
    def train(self):
        task_map = {
            'classification': train_classification,
            # 'detection': train_detection,
            'regression': train_regression,
            'reid': train_reid
        }
        if self.training_task in task_map:
            # print(f"Training {self.training_task} with arguments: {self.kwargs}")
            try:
                return task_map[self.training_task](**self.kwargs)
            except TypeError as e:
                raise TypeError(f"Incorrect arguments for {self.training_task}: {e}")
        else:
            raise ValueError("Invalid function type")
        
    def evaluate(self, test_dataset, task_name):
        task_map = {
            'classification': evaluate_classification,
            # 'detection': evaluate_detection,
            'regression': evaluate_regression,
            'reid': evaluate_reid
        }
        if self.training_task in task_map:
            # Filter out kwargs not used in evaluation
            eval_args = {
                k: v for k, v in self.kwargs.items()
                if k in ['model', 'dataset_name','preprocess','finetune_type','batch_size']  # adjust this list per your actual function
            }
            print(eval_args)
            return task_map[self.training_task](test_dataset=test_dataset, task_name=task_name, **eval_args)
        else:
            raise ValueError("Invalid function type")
