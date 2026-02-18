import numpy as np
from trust.strategies.partition_strategy import PartitionStrategy
from .utils import get_dataset

def load_uids_with_partition_strategy(
    dataset_name,
    model,
    nclasses,
    budget,
    args
):
    """
    Selects indices using PartitionStrategy as a filter.

    Args:
        labeled_dataset: torch.utils.data.Dataset
        unlabeled_dataset: torch.utils.data.Dataset
        model: torch.nn.Module
        nclasses: int
        budget: int
        args: dict (must include 'num_partitions' and 'wrapped_strategy_class')

    Returns:
        np.ndarray: Selected indices from the unlabeled dataset.
    """
    # Set private_dataset and query_dataset to None
    private_dataset = None
    unlabeled_dataset = None
    labeled_dataset = get_dataset(dataset_name, split='train')
    query_dataset = get_dataset(dataset_name, split='val')

    strategy = PartitionStrategy(
        labeled_dataset=labeled_dataset,
        unlabeled_dataset=unlabeled_dataset,
        net=model,
        nclasses=nclasses,
        args=args,
        query_dataset=query_dataset,
        private_dataset=private_dataset
    )
    selected_indices = strategy.select(budget)
    return np.array(selected_indices)