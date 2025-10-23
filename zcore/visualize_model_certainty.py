import argparse
from copy import deepcopy
import IPython
import numpy as np
import os
import torch

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

import core.train as train

def generate_model_predictions(args, pred_f):
    print(f"Generating {pred_f} for {args.dataset}.")

    # Load model.
    checkpoint = torch.load(
        args.model_weights, 
        map_location=args.device,
        weights_only=False
    )
    model = checkpoint["state_dict"].module
    model.to(args.device).eval()

    # Load dataset.
    train_data, test_data = train.load_cifar(args)
    if args.split == "test": current_data = test_data
    elif args.split == "train": current_data = train_data
    else: raise ValueError(f"{args.split} not recognized.")
    data_loader = torch.utils.data.DataLoader(
        current_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    # Generate model outputs and save predictions.
    n_classes = len(data_loader.dataset.classes)
    predictions = np.zeros(shape=(len(current_data), n_classes))
    with torch.no_grad():
        for i, (model_input, target) in enumerate(data_loader):
            if i % 50 == 0: print(f"[{i}/{len(data_loader)}]")
            output = model(model_input.to(args.device))
            idx0, idx1 = i*args.batch_size, (i+1)*args.batch_size
            predictions[idx0:idx1] = output.to("cpu")

    np.save(pred_f, predictions)
    print(f"Predictions saved to {pred_f}")


def main(args):
    
    # Get model predictions.
    exp_name = f"{args.dataset}-{args.split}-predictions"
    pred_f = args.model_weights.replace("checkpoint.pth.tar",f"{exp_name}.npy")
    if not os.path.isfile(pred_f): generate_model_predictions(args, pred_f)
    predictions = np.load(pred_f)
    
    # Generate dataset-wide UMAP surface for model prediction space.
    dataset_fo = foz.load_zoo_dataset(args.dataset, split=args.split)
    fob.compute_visualization(
        dataset_fo,
        embeddings=predictions,
        num_dims=2,
        brain_key=f"model_certainty",
        verbose=True,
        seed=51
    )

    # Calculate and visualize model certainty. 
    n_samples = len(predictions)
    first2second_ratio = np.zeros(n_samples)
    for j in range(n_samples):
        sample_predictions = deepcopy(predictions[j])
        sample_predictions.sort()
        first2second_ratio[j]=sample_predictions[-1]/sample_predictions[-2]
    first2second_score = 1/first2second_ratio    
    dataset_fo.set_values(f"f2s_score", first2second_score)
    session = fo.launch_app(dataset_fo)
    IPython.embed()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize Model Certainty")

    # Model.
    parser.add_argument("--model_weights", type=str)
    parser.add_argument("--device", type=str, default="cpu")

    # Dataset.
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)

    args = parser.parse_args()
    main(args)

