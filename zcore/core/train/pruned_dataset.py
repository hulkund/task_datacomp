import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def load_coreset_dataset(args):
    print(f"Loading {args.prune_rate} pruned {args.dataset}.")
   
    # Load dataset.
    if "cifar" in args.dataset: train_data, test_data = load_cifar(args)
    elif args.dataset == "imagenet": train_data, test_data = load_imagenet(args)
    elif "eurosat" in args.dataset: train_data, test_data = load_eurosat(args)
    else: raise ValueError(f"{args.dataset} not recognized.")

    # Prune dataset.
    score = np.load(args.score_file)
   
    """
    # Convert for TDDS format score and mask.
    score_alt = (score - min(score)) / (max(score) - min(score))
    data_mask = np.load(args.score_file.replace("score.npy", "data_mask.npy"))
    score = np.zeros(n).astype(np.float32)
    for i in range(n): score[i] = score_alt[np.where(data_mask == i)]
    """
    
    if "cifar" in args.dataset:
        train_data.targets = [[t, score[i]] 
                              for i, t in enumerate(train_data.targets)]
    else:
        train_data.samples = [(s[0], [s[1], score[i]]) 
                              for i, s in enumerate(train_data.samples)]
    coreset_mask = np.argsort(score)[int(args.prune_rate * len(train_data)):]
    coreset = torch.utils.data.Subset(train_data, coreset_mask)

    # Dataset loaders.
    train_loader = torch.utils.data.DataLoader(
        coreset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader

def load_imagenet(args):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    path = os.path.join(args.data_dir, args.dataset, "ILSVRC", "Data", "CLS-LOC")
    train_data = datasets.ImageFolder(os.path.join(path, "train"), train_transform)
    test_data = datasets.ImageFolder(os.path.join(path, "val"), test_transform)

    return train_data, test_data

def load_eurosat(args):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # https://github.com/Rumeysakeskin/EuroSat-Satellite-CNN-and-ResNet
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    path = os.path.join(args.data_dir, args.dataset)
    train_data = datasets.ImageFolder(os.path.join(path,"train"), train_transform)
    test_data = datasets.ImageFolder(os.path.join(path,"val"), test_transform)

    return train_data, test_data

def load_cifar(args): 

    if args.dataset == "cifar10":
        mean = [0.4913725490196078, 0.4823529411764706, 0.4466666666666667] 
        std =  [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]
    elif args.dataset == "cifar100":
        mean = [0.5070588235294118, 0.48666666666666664, 0.4407843137254902]
        std = [0.26745098039215687, 0.2564705882352941, 0.27607843137254906]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    if args.dataset == "cifar10":   
        train_data = datasets.CIFAR10(args.data_dir, train=True, 
                                  transform=train_transform)
        test_data = datasets.CIFAR10(args.data_dir, train=False, 
                                 transform=test_transform)
    elif args.dataset == "cifar100":
        train_data = datasets.CIFAR100(args.data_dir, train=True, 
                                  transform=train_transform)
        test_data = datasets.CIFAR100(args.data_dir, train=False, 
                                 transform=test_transform)

    return train_data, test_data
