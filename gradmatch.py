import torch
from DeepCore.deepcore.methods.gradmatch import GradMatch
from DeepCore.deepcore.datasets.cifar10 import CIFAR10
import sys
sys.path.append('/data/vision/beery/scratch/evelyn/task-datacomp/')
from baselines.utils import get_dataset
from baselines.model_backbone import get_model_processor
import argparse

def run_gradmatch_filter(train_split='train', 
                        val_split='val', 
                        fraction=0.5, 
                        random_seed=42, 
                        epochs=200, 
                        balance=True, 
                        lam=1.0,
                        model=None,
                        args=None):
    # Load datasets
    train_dataset = get_dataset('iWildCam', split=train_split)
    val_dataset = get_dataset('iWildCam', split=val_split)
    # look at what type the dataset label target is
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Example data point: {type(train_dataset[0][0])}")
    print(f"Example label type: {type(train_dataset[0][1])}")

    # Prepare model and args (adapt to your setup)
    # Example: args should have .num_classes, .device, .selection_batch, .print_freq, .workers, etc.

    # args.channel = channel
    # args.im_size = im_size
    # args.num_classes = num_classes

    # Initialize GradMatch
    gradmatch = GradMatch(
        dst_train=train_dataset,
        fraction=fraction,
        random_seed=random_seed,
        epochs=epochs,
        specific_model=model,
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

    selected_uids = [train_dataset.data.iloc[i]["uid"] for i in selected_indices]
    return filtered_dataset, selected_indices, selected_weights, selected_uids

# model = get_model_processor("full_finetune_resnet50")
args_dict = {'print_freq': 100, 
             'num_classes': 149, 
             'device': 'cuda', 
             'selection_batch': 4, 
             'workers': 4, 
             'channel':3, 
             'im_size':[224,224],
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
args = argparse.Namespace(**args_dict)
filtered_dataset, selected_indices, selected_weights, selected_uids = run_gradmatch_filter(train_split='train', 
                                                                            val_split='test1', 
                                                                            fraction=0.1, 
                                                                            random_seed=42, 
                                                                            epochs=1, 
                                                                            balance=True, 
                                                                            lam=1.0,
                                                                            args=args,
                                                                            model='ResNet18')
print("selected indices:", selected_indices)
print("selected uids:", selected_uids)

