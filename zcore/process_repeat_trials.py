import argparse
import os

import core.coreset as cs

def main(args):

    if args.save_file == None:
        args.save_file = os.path.join(args.base_score_dir, "results.txt")

    trial_folders, trials = cs.get_trial_list(args.base_score_dir)
    exp_data, set_types = cs.collect_log_data(trial_folders, trials)
    cs.make_experiment_log_table(args, trials, exp_data, set_types)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process Repeat Trials.")
    parser.add_argument("--base_score_dir", type=str)
    parser.add_argument("--save_file", type=str)
    args = parser.parse_args()
    main(args)
