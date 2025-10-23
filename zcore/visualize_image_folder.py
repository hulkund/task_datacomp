import argparse
import os

import IPython
import numpy as np

import fiftyone as fo
import fiftyone.brain as fob

import core.coreset as cs

def main(args):

    # Load dataset and generate model embeddings.
    args.dataset = os.path.basename(args.image_dir)
    dataset = fo.Dataset.from_dir(
        dataset_dir=args.image_dir,
        dataset_type=fo.types.ImageDirectory,
        name=args.dataset,
        )
    model = cs.load_fo_model(args, args.embedding)
    print("Note: embeddings can be calculated separately or saved for resuse.")
    embeddings = dataset.compute_embeddings(model) 

    # Create UMAP visualization from embeddings.
    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        brain_key=args.embedding,
        seed=51,
    )

    # Calculate ZCore score and visualize.
    #scores = cs.zcore_score(args, embeddings) # Scores via embeddings.
    scores = cs.zcore_score(args, results.points) # Scores via UMAP projection.
    dataset.set_values("ZCore Score", scores)
    session = fo.launch_app(dataset)
    IPython.embed()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize Image Folder")

    # Image directory.
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--trial", type=int, default=0)

    # ZCore Parameters (see paper for more details).
    parser.add_argument("--embedding", type=str, default="resnet18", 
                        choices=["resnet18", "clip", "dinov2"])
    parser.add_argument("--n_sample", type=int, default=int(1e6))
    parser.add_argument("--sample_dim", type=int, default=2)
    parser.add_argument("--no_rand_init", dest="rand_init", 
                        action="store_false", default=True)
    parser.add_argument("--redund_exp", type=int, default=2)
    parser.add_argument("--redund_nn", type=int, default=10)

    args = parser.parse_args()
    main(args)

