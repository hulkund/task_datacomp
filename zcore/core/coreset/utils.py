import glob
import numpy as np
import os

def experiment_name(args):

    exp_name = "zcore"
    exp_name += f"-{args.dataset}"
    for m in args.embedding: exp_name += f"-{m}"
    exp_name += f"-{int(args.n_sample/1000)}Ks"
    exp_name += f"-{args.sample_dim}sd"
    if args.rand_init: exp_name += "-ri"
    exp_name += f"-{args.redund_nn}nn"
    exp_name += f"-{args.redund_exp}ex"
    exp_name += f"-{args.trial}"

    exp_file = os.path.join(args.results_dir, 
        args.dataset, 
        exp_name, 
        "score.npy"
    )
    os.makedirs(os.path.dirname(exp_file), exist_ok=True)

    return exp_name, exp_file

def get_trial_list(base_score_dir):
    trial_folders = sorted(glob.glob(base_score_dir + "-*"))
    trials = [f.split(base_score_dir + "-")[-1] for f in trial_folders]
    return trial_folders, trials

def collect_log_data(trial_folders, trials):

    exp_data = {}
    set_types = []

    for i, t in enumerate(trials):

        log_folders = sorted(glob.glob(os.path.join(trial_folders[i],"p*")))
        logs = [os.path.join(f,"train_log.txt") for f in log_folders]
        acc = [log_validation_accuracy(l) for l in logs]

        set_names = [os.path.basename(f) for f in log_folders]
        exp_data[t] = {}
        for j, s in enumerate(set_names):
            exp_data[t][s] = acc[j]

        set_types = sorted(list(set(set_types) | set(set_names)))

    return exp_data, set_types

def log_validation_accuracy(file_name):
    try:
        lines = read_text_file(file_name)
        acc = float(lines[-1].split("Accuracy=")[-1].split(", Error=")[0])
    except:
        print(f"{file_name} incomplete.")
        acc = "NA"
    return acc

def make_experiment_log_table(args, trials, exp_data, set_types):

    heading = f"Score: {args.base_score_dir}\n{'Setting' : <8}"
    all_results = {}
    for s in set_types: 
        heading += f"{s : <8}"
        all_results[s] = []
    print_statements(args.save_file, heading, make_dir=True)

    print_statements(args.save_file, "\nTrial Results")
    for t in trials:
        result = f"{t : <8}"
        for s in set_types:
            try:
                r = exp_data[t][s]
                result += f"{r : <8.2f}"
                all_results[s].append(r)
            except: 
                result += f"{'NA' : <8}"

        print_statements(args.save_file, result)
 
    print_statements(args.save_file, "\nAggregate Results")
    mean_result = f"{'Mean' : <8}"
    stddev_result = f"{'StdDev' : <8}"
    overall_mean = []
    for s in set_types:
        overall_mean.append(np.mean(all_results[s]))
        mean_result += f"{overall_mean[-1] : <8.2f}"
        stddev_result += f"{np.std(all_results[s]) : <8.3f}"
    print_statements(args.save_file, mean_result)
    print_statements(args.save_file, stddev_result)
    print_statements(args.save_file, 
                     f"Overall Mean: {np.mean(overall_mean) : <8.2f}\n")

    print(f"\nProcessed log table saved at {args.save_file}")

def read_text_file(file_name, remove_return=True):
    with open(file_name, "r") as f:
        lines = f.readlines()
    if remove_return:
        lines = [l.split("\n")[0] for l in lines]
    return lines

def print_statements(file_name, statements, add_return=True, make_dir=False):

    if isinstance(statements, str):
        statements = [statements]
    if make_dir:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    output_file = open(file_name, "a")

    for statement in statements:
        print(statement)
        if add_return:
            statement = f"{str(statement)}\n"
        output_file.write(statement)

    output_file.close()
