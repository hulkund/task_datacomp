import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from baselines.utils import get_dataset
import clip
from transformers import AutoImageProcessor, AutoModel
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_names = ["ResNet50", "CLIP", "DINOv2"]


def get_resnet50_embedder():
    model = models.resnet50(pretrained=True).to(device).eval()
    model.fc = torch.nn.Identity()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform

def get_clip_embedder():
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

def get_dinov2_embedder():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    return model, processor

def extract_embeddings(dataloader, batch_size=32, out_dir="embeddings", prefix="dataset"):
    resnet_model, resnet_transform = get_resnet50_embedder()
    clip_model, clip_preprocess = get_clip_embedder()
    dinov2_model, dinov2_processor = get_dinov2_embedder()

    resnet_embeds, clip_embeds, dinov2_embeds = [], [], []
    for batch in dataloader:
        img, _, _, _ = batch
        imgs = [transforms.ToPILImage()(im) for im in img]

        # ResNet50
        resnet_batch = torch.stack([resnet_transform(img) for img in imgs]).to(device)
        with torch.no_grad():
            resnet_out = resnet_model(resnet_batch)
        resnet_embeds.append(resnet_out.cpu().numpy())

        # CLIP
        clip_batch = torch.stack([clip_preprocess(img) for img in imgs]).to(device)
        with torch.no_grad():
            clip_out = clip_model.encode_image(clip_batch)
        clip_embeds.append(clip_out.cpu().numpy())

        # DINOv2
        dinov2_batch = [dinov2_processor(images=img, return_tensors="pt").to(device) for img in imgs]
        with torch.no_grad():
            dinov2_out = [dinov2_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy() for inputs in dinov2_batch]
        dinov2_embeds.append(np.concatenate(dinov2_out, axis=0))

    resnet_embeds = np.concatenate(resnet_embeds, axis=0)
    clip_embeds = np.concatenate(clip_embeds, axis=0)
    dinov2_embeds = np.concatenate(dinov2_embeds, axis=0)
    np.save(os.path.join(out_dir, f"{prefix}_resnet50.npy"), resnet_embeds)
    np.save(os.path.join(out_dir, f"{prefix}_clip.npy"), clip_embeds)
    np.save(os.path.join(out_dir, f"{prefix}_dinov2.npy"), dinov2_embeds)
    return resnet_embeds, clip_embeds, dinov2_embeds

# def save_embeddings(embeddings, out_dir, prefix):
#     os.makedirs(out_dir, exist_ok=True)

def compute_kl_wasserstein(emb1, emb2):
    # KL divergence (between mean distributions)
    p = np.mean(emb1, axis=0)
    q = np.mean(emb2, axis=0)
    # Add small value to avoid log(0)
    p = np.clip(p, 1e-8, None)
    q = np.clip(q, 1e-8, None)
    kl = entropy(p, q)
    # Wasserstein distance (flattened)
    ot = wasserstein_distance(emb1.flatten(), emb2.flatten())
    return kl, ot

def get_perf_dict(subset_file, test_num):
    acc, subset_size, emb_dist_dict = get_perf(subset_file, test_num)
    return {
        'subset_file': subset_file,
        'test_num': test_num,
        'acc': acc,
        'subset_size': subset_size,
        'ResNet50_OT': emb_dist_dict['ResNet50']['Wasserstein'],
        'ResNet50_KL': emb_dist_dict['ResNet50']['KL'],
        'CLIP_OT': emb_dist_dict['CLIP']['Wasserstein'],
        'CLIP_KL': emb_dist_dict['CLIP']['KL'],
        'DINOv2_OT': emb_dist_dict['DINOv2']['Wasserstein'],
        'DINOv2_KL': emb_dist_dict['DINOv2']['KL'],
    }

def get_perf(subset_file, test_num):
    initial_directory = "/".join(subset_file.split("/")[:3])
    # getting datasets
    train_subset = get_dataset(dataset_name='iWildCam', split='train', subset_path=subset_file)
    test_set = get_dataset('iWildCam', split=test_num)
    train_subset_dl = DataLoader(dataset=train_subset, batch_size=16, shuffle=True, num_workers=4)
    test_dl = DataLoader(dataset=test_set, batch_size=16, shuffle=True, num_workers=4)
    # getting distances
    ds_name1 = subset_file.split('/')[-1].replace('.npy','')
    ds_name2 = test_num
    out_dir = f"{initial_directory}/embeddings/"
    emb_dist_dict = get_distance(train_subset_dl, test_dl, ds_name1, ds_name2, out_dir)
    return emb_dist_dict

def parse_performance(file_path):
    # Implement this function based on your file structure
    # For this example, let's assume each file contains a single performance metric value
    with open(file_path, 'r') as file:
        # print(file_path)
        data=json.load(file)
        try:
            acc = data.get('acc',None)
            subset_size = data.get('subset_size',None)
        except:
            return None, None
    return acc, subset_size

def embdist_vs_performance(title):
    """ Computes embedding distances vs performance and saves to CSV """
    directory = "experiments_again/iWildCam/"
    subset_files = glob.glob(os.path.join(directory, "**", "*subset.npy"), recursive=True)
    dict_subset_files = []
    for i in range(len(subset_files)):
        subset_file = subset_files[i]
        test_num = subset_file.split('/')[3][:5]
        # getting performance and distance if its all_s then do for all test nums
        if test_num == "all_s":
            for test_num in ["test1", "test2", "test3", "test4"]:
                dist_dict = get_perf_dict(subset_file, test_num)
                dict_subset_files.append(dist_dict)
        # otherwise just do for that test num
        else:
            print(subset_file)
            dist_dict = get_perf_dict(subset_file, test_num)
            dict_subset_files.append(dist_dict)
    # make dataframe
    df = pd.DataFrame(dict_subset_files)
    # save dataframe
    df.to_csv(f"{title}.csv", index=False)
    return df

def get_distance(ds1, ds2 , ds_name1, ds_name2, out_dir, batch_size=32):
    # Extract and save embeddings
    emb1 = extract_embeddings(ds1, batch_size, out_dir, ds_name1)
    emb2 = extract_embeddings(ds2, batch_size, out_dir, ds_name2)

    embedding_names = ["ResNet50", "CLIP", "DINOv2"]
    distances = {}
    # Compute distances for each model
    for name, e1, e2 in zip(embedding_names, emb1, emb2):
        kl, ot = compute_kl_wasserstein(e1, e2)
        distances[name] = {'KL': kl, 'Wasserstein': ot}
        print(f"{name} - KL Divergence: {kl:.4f}, Wasserstein Distance: {ot:.4f}")
    return distances

def plot_performance_vs_distance(df,title):
    # plot distances vs accuracies for all three embeddings and both OT and KL in a 3x2 grid with a correlation coefficient in the title
    metrics = ['OT', 'KL']
    embeddings = ['ResNet50', 'CLIP', 'DINOv2']
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    for i, emb in enumerate(embeddings):        
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            col_name = f"{emb}_{metric}"
            df[col_name] = df[col_name].astype(float)
            correlation_matrix = np.corrcoef(df[col_name], df['acc'])
            correlation_coefficient = correlation_matrix[0, 1]
            title = title+f"_{emb}_{metric}"
            sns.scatterplot(x=df[col_name], y=df['acc'], ax=ax)
            ax.set_xlabel(f'{emb} {metric} Distance')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{emb} {metric} vs Accuracy (Corr: {correlation_coefficient:.2f})')
            ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()   
    

if __name__ == "__main__":
    # otdd_vs_performance()
    title= "iwildcam_embdist_vs_performance"
    df = embdist_vs_performance(title)
    plot_performance_vs_distance(df,title)


