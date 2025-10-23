import argparse
import numpy as np
import os

import core.coreset as cs

def main(args):

    exp_name, exp_file = cs.experiment_name(args)
    assert not os.path.exists(exp_file), f"{exp_file} already exists."

    embeddings = cs.get_model_embedding(args)
    scores = cs.zcore_score(args, embeddings)
    np.save(exp_file, scores)
    print(f"\nZCore score saved at {exp_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Zero-Shot Coreset Selection (ZCore)")

    parser.add_argument("--trial", type=int, default=0)

    # Dataset.
    dataset_choice = ["cifar10", "cifar100", "imagenet", "eurosat10", 
                      "eurosat20", "eurosat40", "eurosat80"]
    parser.add_argument("--dataset", type=str, choices=dataset_choice)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=2)

    # ZCore Parameters (see paper for more details).
    parser.add_argument("--embedding", type=str, nargs="+", 
                        choices=["resnet18", "clip", "dinov2"])
    parser.add_argument("--n_sample", type=int, default=int(1e6))
    parser.add_argument("--sample_dim", type=int, default=2)
    parser.add_argument("--no_rand_init", dest="rand_init", 
                        action="store_false", default=True)
    parser.add_argument("--redund_exp", type=int, default=4)
    parser.add_argument("--redund_nn", type=int, default=1000)

    args = parser.parse_args()
    main(args)

