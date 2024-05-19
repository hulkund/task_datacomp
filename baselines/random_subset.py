import argparse
import numpy as np
import pandas as pd 
import torch 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset_size",
        default=1000,
        type=int
    )
    args = parser.parse_args()
    