import os
from main_scripts.utils import create_save_folder, create_sweep_dict, get_sweep_combinations

def _recreate_pt_path(save_folder, test_split, finetune_type, lr, batch_size):
    pt_path = f"{save_folder}{test_split}_{finetune_type}_lr={lr}_batchsize={batch_size}_logits.pt"
    return pt_path

def get_pt_paths(baselines_list, dataset_list, finetune_list, lr_list, batch_size_list):
    sweep_dict = create_sweep_dict()
    pt_paths = []

    for baseline in baselines_list:
        print("="*50)
        params = sweep_dict[baseline]
        for param_setting in get_sweep_combinations(params, baseline):
            print("Getting pt_path for param configuration:", param_setting)

            for dataset, val_split, test_split in dataset_list:
                for finetune_type in finetune_list:
                    if finetune_type=="linear_probe": lr_list = [0]
                    else: lr_list = lr_list
                    for lr in lr_list:
                        for batch_size in batch_size_list:
                            save_folder = create_save_folder(dataset, baseline, param_setting)
                            pt_path = _recreate_pt_path(save_folder, test_split, finetune_type, lr, batch_size)
                            if os.path.exists(pt_path):
                                pt_paths.append(pt_path)
                            else:
                                print("Warning: Path does not exist:", pt_path)
    return pt_paths

def create_confusion_matrix_png_save_path(pt_path):
    png_save_path = os.path.splitext(pt_path)[0] + "_confusion_matrix.png"
    return png_save_path

def create_per_class_accuracy_json_save_path(pt_path):
    json_save_path = os.path.splitext(pt_path)[0] + "_class_accuracy.json"
    return json_save_path

def get_per_class_json_paths(baselines_list, dataset_list, finetune_list, lr_list, batch_size_list):
    pt_paths = get_pt_paths(baselines_list, dataset_list, finetune_list, lr_list, batch_size_list)

    json_paths = []
    for pt_path in pt_paths:
        json_path = create_per_class_accuracy_json_save_path(pt_path)
        if os.path.exists(json_path):
            json_paths.append(json_path)
        else:
            print("Warning: Path does not exist:", json_path)
    return json_paths