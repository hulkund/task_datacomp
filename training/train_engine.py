import numpy as np
import torch.nn as nn
import inspect
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from training.train_on_subset_classification import train_classification, evaluate_classification
from training.train_on_subset_regression import train_regression, evaluate_regression
# from training.train_on_subset_reid import train_reid, evaluate_reid

class TrainEngine():
    def __init__(self, training_task, **kwargs):
        self.training_task = training_task
        self.kwargs = kwargs
        
    def train(self):
        task_map = {
            'classification': train_classification,
            # 'detection': train_detection,
            'regression': train_regression,
            # 'reid': train_reid
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
            # 'reid': evaluate_reid
        }
        if self.training_task in task_map:
            eval_fn = task_map[self.training_task]
            allowed_params = set(inspect.signature(eval_fn).parameters.keys())

            # Pass only kwargs accepted by the selected evaluator.
            eval_args = {
                k: v for k, v in self.kwargs.items()
                if k in allowed_params
            }

            result = eval_fn(test_dataset=test_dataset, task_name=task_name, **eval_args)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            # Backward compatibility for evaluators that return only metrics.
            return result, {"logits": None, "labels": None, "predictions": None}
        else:
            raise ValueError("Invalid function type")
