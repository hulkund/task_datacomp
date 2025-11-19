from DeepCore.deepcore.methods.gradmatch import GradMatch
import argparse
import numpy as np
from pathlib import Path
from baselines.utils import get_dataset
from typing import Union
import torch

def load_uids_with_gradmatch(
    fraction=0.25, # sweep over [0.25, 0.5, 0.75, 0.9]
    random_seed=42, 
    epochs=50, # paper uses 200, but we've found that finetuned ResNet overfits on iWildCam after 50 epochs
    balance=True, 
    lam=0.5, # from the paper
    args=None
) -> np.ndarray:
    # TODO: put a bunch of the args_dict into config (or hard-code them here)
    args_dict = {
        'print_freq': 100, 
        'device': 'cuda', 
        'workers': 4, 
        'model':'ResNet18', # from the paper
        'selection_optimizer':'SGD', # from the paper
        'selection_momentum':0.9, # from the paper
        'selection_weight_decay':1e-4, # from the paper
        'selection_nesterov':True,
        'selection_test_interval':10,
        'selection_test_fraction':1.0,
        'specific_model':None,
        'torchvision_pretrain':True,
        'if_dst_pretrain':False,
        'dst_pretrain_dict':{},
        'n_pretrain_size':1000,
        'n_pretrain':1000,
        'gpu':[0]
    }

    # Load datasets using args.dataset_name and args.val_embedding_path
    p = Path(args.val_embedding_path) # something like 'all_datasets/iWildCam/embeddings/val1_embeddings.npy'
    filename = p.name
    dataset_dir = p.parent.parent.name
    dataset_name = dataset_dir
    val_split = filename.split('_')[0]
    print(f"dataset={dataset_name}, val_split={val_split}")
    train_dataset = get_dataset(dataset_name, split='train')
    val_dataset = get_dataset(dataset_name, split=val_split)
    
    # look at what type the dataset label target is
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Example data point: {type(train_dataset[0][0])}")
    print(f"Example label type: {type(train_dataset[0][1])}")

    # TODO: determine these values from train_dataset
    args_dict['num_classes'] = train_dataset.data['label'].nunique()
    args_dict['channel'] = 3
    args_dict['im_size'] = [224,224]

    # Prepare model and args (adapt to your setup)
    # Example: args should have .num_classes, .device, .selection_batch, .print_freq, .workers, etc.

    user_args = vars(args)

    # TODO: can probably cross-reference with config file
    if 'selection_lr' in user_args:
        user_args['selection_lr'] = float(user_args['selection_lr'])
    if 'selection_batch' in user_args:
        user_args['selection_batch'] = int(user_args['selection_batch'])

    merged_dict = {**user_args, **args_dict}
    args = argparse.Namespace(**merged_dict)

    print("Initializing GradMath with args:", vars(args))

    # Initialize GradMatch
    gradmatch = GradMatch(
        dst_train=train_dataset,
        fraction=fraction,
        random_seed=random_seed,
        epochs=epochs,
        balance=balance,
        dst_val=val_dataset,
        lam=lam,
        args=args
    )
    print(gradmatch)
    # Run selection
    selection_result = gradmatch.select()
    selected_indices = selection_result["indices"]
    selected_weights = selection_result["weights"]

    # Filter your dataset using selected_indices
    filtered_dataset = torch.utils.data.Subset(train_dataset, selected_indices)
    print(f"Selected {len(selected_indices)} samples using GradMatch.")

    # Optionally, save indices or filtered dataset
    # np.save("gradmatch_selected_indices.npy", selected_indices)

    selected_uids = np.array([train_dataset.data.iloc[i]["uid"] for i in selected_indices])
    # return filtered_dataset, selected_indices, selected_weights, selected_uids
    return selected_uids
