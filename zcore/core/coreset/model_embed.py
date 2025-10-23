import fiftyone as fo
import fiftyone.zoo as foz
import numpy as np
import os
import pickle

def load_fo_dataset(args):
    
    if "cifar" in args.dataset:
        dataset = foz.load_zoo_dataset(args.dataset, split="train")
    else: 
    
        if args.dataset == "imagenet":
            path = os.path.join(args.data_dir, "imagenet", "ILSVRC", "Data", 
                                "CLS-LOC", "train")
        elif "eurosat" in args.dataset:
            path = os.path.join(args.data_dir, args.dataset, "train")
        else:
            raise ValueError(f"{args.dataset} not recognized.")

        dataset = fo.Dataset.from_dir(
            path, 
            dataset_type=fo.types.ImageClassificationDirectoryTree
        )

    return dataset

def load_fo_model(args, model_name):

    if model_name == "clip":
        model = foz.load_zoo_model("open-clip-torch",
            clip_model="ViT-L-14",
            pretrained="openai",
        )
    elif model_name == "resnet18":
        model = foz.load_zoo_model("resnet18-imagenet-torch")
    elif model_name == "dinov2":
        model = foz.load_zoo_model("dinov2-vitb14-torch")
    else: model = foz.load_zoo_model(model_name)
    
    return model

def generate_embedding(args, model_name, embed_file):

    dataset = load_fo_dataset(args)
    model = load_fo_model(args, model_name)

    print(f"Generating {args.dataset}-{model_name} embeddings.")
    model_embed = dataset.compute_embeddings(model)
    os.makedirs(os.path.dirname(embed_file), exist_ok=True)
    pickle.dump(model_embed, open(embed_file, "wb"))
    print(f"Model embeddings saved at {embed_file}.")
   
    return model_embed

def get_model_embedding(args):

    embed_dir = os.path.join(args.data_dir, "preprocess", args.dataset)
    
    for model_name in args.embedding:
        
        embed_file = os.path.join(embed_dir, f"{model_name}_embedding.pk")
        if os.path.exists(embed_file):
            model_embed = pickle.load(open(embed_file, "rb"))
        else:
            model_embed = generate_embedding(args, model_name, embed_file)

        if "total_embed" in locals():
            total_embed = np.concatenate((total_embed, model_embed), axis=1)
        else: total_embed = model_embed

    return total_embed
