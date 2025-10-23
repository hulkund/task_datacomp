import argparse
import core.train as t

def main(args):

    t.seed_everything(args.manual_seed)
    train_loader, test_loader = t.load_coreset_dataset(args)
    model = t.load_model(args, len(test_loader.dataset.classes))
    t.train_coreset_model(args, model, train_loader, test_loader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Coreset Model")

    parser.add_argument("--manual_seed", type=int, default=51)

    # Dataset.
    dataset_choice = ["cifar10", "cifar100", "imagenet", "eurosat10", 
                      "eurosat20", "eurosat40", "eurosat80"]
    parser.add_argument("--dataset", type=str, choices=dataset_choice)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cpu", "mps", "cuda"])

    # Model.
    parser.add_argument("--architecture", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--decay", type=float, default=0.0005)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--print_freq", type=int, default=200)

    # Coreset.
    parser.add_argument("--score_file", type=str)
    parser.add_argument("--prune_rate", type=float, default=0.7)

    args = parser.parse_args()
    main(args)
