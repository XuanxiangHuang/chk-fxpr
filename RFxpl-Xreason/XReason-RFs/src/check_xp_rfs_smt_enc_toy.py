from __future__ import print_function
from anchor_wrap import anchor_call
from data import Data
from options import Options
import argparse
import os
import resource
import sys
from xgbooster import XGBooster

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check explanations from PyXAI RFS (toy datasets)')
    parser.add_argument('-x', '--xtype', type=str, required=True,
                        choices=['abd', 'con'],
                        help="Explanation type: 'abd' for AXp or 'con' for CXp")
    args = parser.parse_args()

    xtype = args.xtype
    data_list = "datasets_small.list"

    verbose = False

    datasets = []
    depths = []
    nums = []

    # Read the file
    with open(data_list, "r") as file:
        for line in file:
            parts = line.strip().split()  # Split by spaces
            dataset_path = parts[0]  # First part is the dataset path
            depth = int(parts[1])  # Second part is the depth (converted to int)
            num = int(parts[2])  # Third part is the number of estimators (converted to int)

            datasets.append(dataset_path)
            depths.append(depth)
            nums.append(num)

    for data, depth, num in zip(datasets, depths, nums):
        base = os.path.splitext(os.path.basename(data))[0]
        enc_path = f"smt_enc/Classifiers/{base}_nbestim_{num}_maxdepth_{depth}.smt"
        opts = Options(None)
        opts.n_estimators = num
        opts.files = enc_path

        data = Data(filename=data, mapfile=opts.mapfile,
                    separator=opts.separator,
                    use_categorical=opts.use_categorical)
        xgb = XGBooster(opts, from_data=data)
        xgb.__init__(opts, from_encoding=opts.files)

        i_list = []
        expl_list = []

        log_path = f"pyxai_results/rfs/toy/{xtype}/{base}.{num}.log"
        xp_label = "AXp" if xtype == 'abd' else "CXp"

        try:
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("i:"):
                        values = list(map(float, line[2:].strip().split(',')))
                        i_list.append(values)
                    elif line.startswith("expl:"):
                        values = list(map(int, line[5:].strip().strip('[]').split(',')))
                        expl_list.append(values)
        except FileNotFoundError:
            print(f"File not found: {log_path} — skipping.")
            continue

        tested = set()
        invalid_xp = 0
        valid_xp = 0
        num_wxp = 0
        output_str = ''

        print(base, depth, num)

        for i, (inst, expl) in enumerate(zip(i_list, expl_list)):
            opts.explain = inst
            if tuple(opts.explain) in tested:
                continue
            tested.add(tuple(opts.explain))

            # validating the explanation
            expl = [i-1 for i in expl]
            valid, real_xp = xgb.validate(opts.explain, expl, exp_type=xtype)
            if valid:
                valid_xp += 1
            elif len(real_xp) == 0:
                invalid_xp += 1
            else:
                num_wxp += 1

            output_str += f"i: {','.join(f'{x}' for x in inst)}\n"
            output_str += f"PyXAI: {expl}\n"
            output_str += f"{xp_label}: {real_xp}\n"
            output_str += f"Redundant %: {(len(expl) - len(real_xp)) * 100 / len(expl):.1f}\n\n"

        output_str += f"nof. valid: {valid_xp}\n"
        output_str += f"nof. invalid: {invalid_xp}\n"
        output_str += f"nof. non-minimal: {num_wxp}\n"
        open(f"results/pyxai_vs_xreason/toy/{xtype}/{base}.{num}.log", 'w').write(output_str)
