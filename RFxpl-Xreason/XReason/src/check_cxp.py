from __future__ import print_function
from anchor_wrap import anchor_call
from data import Data
from options import Options
import os
import resource
import sys
from xgbooster import XGBooster

if __name__ == '__main__':
    data_list = "datasets.list"

    num = 50
    verbose = False

    datasets = []
    depths = []

    # Read the file
    with open(data_list, "r") as file:
        for line in file:
            parts = line.strip().split()  # Split by spaces
            dataset_path = parts[0]  # First part is the dataset path
            depth = int(parts[1])  # Second part is the depth (converted to int)

            datasets.append(dataset_path)
            depths.append(depth)

    for data, depth in zip(datasets, depths):
        base = os.path.splitext(os.path.basename(data))[0]
        pickle_path = f'../models/{base}/{base}_nbestim_{num}_maxdepth_{depth}_testsplit_0.2.mod.pkl'

        opts = Options(None)
        opts.n_estimators = num
        opts.files = pickle_path
        xgb = XGBooster(opts, from_model=opts.files)

        # encode it and save the encoding to another file
        xgb.encode()

        i_list = []
        expl_list = []

        log_path = f"pyxai_results/con/{base}.{num}.log"

        try:
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("i:"):
                        values = list(map(float, line[2:].strip().split(',')))
                        i_list.append(values)
                    elif line.startswith("expl:"):
                        # strip off “expl:” and surrounding brackets
                        raw = line[5:].strip().strip('[]')
                        # if raw is non-empty, split & map; otherwise give []
                        values = list(map(int, raw.split(','))) if raw else []
                        expl_list.append(values)
        except FileNotFoundError:
            print(f"File not found: {log_path} — skipping.")
            continue

        tested = set()
        invalid_xp = 0
        valid_xp = 0
        num_wxp = 0
        output_str = ''

        for i, (inst, expl) in enumerate(zip(i_list, expl_list)):
            opts.explain = inst
            if tuple(opts.explain) in tested:
                continue
            tested.add(tuple(opts.explain))

            if expl:
                expl = [i-1 for i in expl]
                valid, real_cxp = xgb.validate(opts.explain, expl, exp_type='con')
                if valid:
                    valid_xp += 1
                elif len(real_cxp) == 0:
                    invalid_xp += 1
                else:
                    num_wxp += 1
            else:
                real_cxp = []
                invalid_xp += 1

            output_str += f"i: {','.join(f'{x}' for x in inst)}\n"
            output_str += f"PyXAI: {expl}\n"
            output_str += f"CXp: {real_cxp}\n"
            if len(expl) == 0:
                output_str += "Redundant %: NA\n\n"
            else:
                output_str += f"Redundant %: {(len(expl) - len(real_cxp)) * 100 / len(expl):.1f}\n\n"

        output_str += f"nof. valid: {valid_xp}\n"
        output_str += f"nof. invalid: {invalid_xp}\n"
        output_str += f"nof. non-minimal: {num_wxp}\n"
        open(f"results/pyxai_vs_xreason/con/{base}.{num}.log", 'w').write(output_str)
