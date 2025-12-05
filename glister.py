import numpy as np
import torch
from DeepCore.deepcore.methods.glister import Glister
import sys
sys.path.append('/data/vision/beery/scratch/evelyn/task-datacomp/')
from baselines.utils import get_dataset
import argparse

def run_glister_filter(train_split='train', 
                        val_split='val', 
                        fraction=0.5, 
                        random_seed=42, 
                        epochs=1, 
                        balance=True, 
                        eta=0.1,
                        model=None,
                        args=None):
    # Load datasets
    train_dataset = get_dataset('iWildCam', split=train_split)
    val_dataset = get_dataset('iWildCam', split=val_split)
    
    print(f"Before: len(train_dataset.data)={len(train_dataset.data)}, train_dataset.classes={train_dataset.classes}")
    print(f"Before: len(val_dataset.data)={len(val_dataset.data)}, val_dataset.classes={val_dataset.classes}")
    
    val_labels = set(np.unique(val_dataset.labels))
    print(f"{val_labels = }")

    train_indices = train_dataset.labels[train_dataset.labels.isin(val_labels)].index

    label_map = {old: new for new, old in enumerate(sorted(val_labels))}
    print(f"{label_map = }")

    filtered_train_df = train_dataset.data.iloc[train_indices].reset_index(drop=True)
    filtered_train_df["label"] = filtered_train_df["label"].map(label_map)
    
    filtered_val_df = val_dataset.data
    filtered_val_df["label"] = filtered_val_df["label"].map(label_map)

    # making new train and val datasets
    train_dataset = get_dataset('iWildCam', split='train', dataframe=filtered_train_df)
    val_dataset = get_dataset('iWildCam', split=val_split, dataframe=filtered_val_df)

    print(f"After: len(train_dataset.data)={len(train_dataset.data)}, train_dataset.classes={train_dataset.classes}")
    print(f"After: len(val_dataset.data)={len(val_dataset.data)}, val_dataset.classes={val_dataset.classes}")

    # Look at what type the dataset label target is
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Example data point: {type(train_dataset[0][0])}")
    print(f"Example label type: {type(train_dataset[0][1])}")

    args_dict = {'print_freq': 100, 
                'device': 'cuda', 
                'selection_batch': 4, 
                'workers': 4, 
                'model':'ResNet18',
                'selection_optimizer':'SGD',
                'selection_lr':0.01,
                'selection_momentum':0.9,
                'selection_weight_decay':1e-4,
                'selection_nesterov':True,
                'selection_test_interval':10,
                'selection_test_fraction':1.0,
                'specific_model':None,
                'torchvision_pretrain':True,
                'if_dst_pretrain':False,
                'dst_pretrain_dict':{},
                'n_pretrain_size':1000,
                'n_pretrain':1000,
                'gpu':[0]}
    args_dict['num_classes'] = train_dataset.data['label'].nunique()
    args_dict['channel'] = 3
    args_dict['im_size'] = [224,224]
    args = argparse.Namespace(**args_dict)

    # Initialize Glister
    glister = Glister(
        dst_train=train_dataset,
        fraction=fraction,
        random_seed=random_seed,
        eta=eta,
        epochs=epochs,
        specific_model=model,
        balance=balance,
        dst_val=val_dataset,
        args=args
    )
    print(glister)
    # Run selection
    selection_result = glister.select()
    selected_indices = selection_result["indices"]

    # Filter your dataset using selected_indices
    filtered_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
    print(f"Selected {len(selected_indices)} samples using Glister.")

    selected_uids = [train_dataset.data.iloc[i]["uid"] for i in selected_indices]
    return filtered_dataset, selected_indices, selected_uids

filtered_dataset, selected_indices, selected_uids = run_glister_filter(train_split='train', 
                                                                       val_split='val1', 
                                                                       fraction=0.1, 
                                                                       random_seed=42, 
                                                                       epochs=1, 
                                                                       balance=True, 
                                                                       eta=0.1,
                                                                       model='ResNet18')
print("selected indices:", selected_indices)
print("selected uids:", selected_uids)
