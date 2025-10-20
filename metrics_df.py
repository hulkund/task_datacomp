import numpy as np
import pandas as pd
import os
import json

def parse_performance(file_path):
    # Implement this function based on your file structure
    # For this example, let's assume each file contains a single performance metric value
    with open(file_path, 'r') as file:
        print(file_path)
        data=json.load(file)
        try:
            performance = data.get('acc',None)
            precision = data.get('precision',None)
            recall = data.get('recall',None)
        except:
            return None, None, None
    return performance, precision, recall

def parse_mse(file_path):
    with open(file_path, 'r') as file:
        print(file_path)
        data=json.load(file)
        performance = data.get('mse',None)
    return performance, None, None

def find_best_hyperparameters(base_path):
    data = []
    for training_dataset in ["iWildCam"]:#os.listdir(base_path):
        training_dataset_path = os.path.join(base_path, training_dataset)
        print("training_dataset_path:", training_dataset_path)
        if os.path.isdir(training_dataset_path):
            for baseline in os.listdir(training_dataset_path):
                baseline_path = os.path.join(training_dataset_path, baseline)
                if os.path.isdir(baseline_path):
                    best_performance = {}
                    for file_name in os.listdir(baseline_path):
                        file_path = os.path.join(baseline_path, file_name)
                        if os.path.isfile(file_path) and '.json' in file_name:
                            performance, precision, recall = parse_performance(file_path)
                            if performance is not None:
                                test_set, hyperparam = file_name.split('_', 1)
                                if test_set not in best_performance or performance > best_performance[test_set][1]:
                                    best_performance[test_set] = (hyperparam, performance, precision, recall)
                    
                    for test_set, (hyperparam, performance, precision, recall) in best_performance.items():
                        data.append({
                            'Training Dataset': training_dataset,
                            'Baseline': baseline,
                            'Test Set': test_set,
                            # 'Best Hyperparameter': hyperparam,
                            'Performance': performance,
                            'Precision': precision, 
                            'Recall': recall
                        })

    df = pd.DataFrame(data)
    return df

def pivot_data(df):
    # Create a new column name for each combination of training dataset and test set
    column_names = [(dataset, test_set) for dataset in df['Training Dataset'].unique() for test_set in sorted(df['Test Set'].unique())]
    columns = pd.MultiIndex.from_tuples(column_names, names=['Training Dataset', 'Test Set'])
    
    # Create a pivot table with the new column names
    df_pivot = df.pivot_table(index=['Baseline'],#, 'Best Hyperparameter'],
                              columns=['Training Dataset', 'Test Set'],
                              values='Performance',
                              aggfunc='first').reindex(columns=columns)
    return df_pivot

# Example usage
base_path = os.getcwd()
print("base_path:", base_path)
before_df = find_best_hyperparameters(base_path + "/experiments")
# print("before:")
# print(before_df)
df = pivot_data(before_df)
# print("After:")
print(df)