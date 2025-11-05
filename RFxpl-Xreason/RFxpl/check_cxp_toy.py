from __future__ import print_function
from data import Data
from options import Options
import os
import resource
import sys
from xrf import XRF
from xrf import RFSklearn
import numpy as np
import pandas as pd

if __name__ == '__main__':
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
        df = pd.read_csv(f"../bench/{base}/{base}.csv")
        # Drop the last column (output column)
        df_X = df.iloc[:, :-1]
        # Extract min and max values for each feature
        min_values = df_X.min()
        max_values = df_X.max()
        # Create a dictionary with feature index as key
        min_max_dict = {i: {'min': min_values.iloc[i], 'max': max_values.iloc[i]} for i in range(df_X.shape[1])}
        pickle_path = f'Classifiers/{base}/{base}_nbestim_{num}_maxdepth_{depth}.mod.pkl'

        opts = Options(None)
        opts.n_estimators = num
        opts.files = pickle_path

        rf_md = RFSklearn(from_file=opts.files)
        feature_names, target_name = rf_md.feature_names, rf_md.targets
        rf_exp = XRF(rf_md, feature_names, target_name)

        i_list = []
        i_pred_list = []
        expl_list = []
        pyxai_wit_list = []
        pyxai_pred_wit_list = []

        log_path = f"pyxai_results/toy/con/{base}.{num}.log"

        try:
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("i:"):
                        values = list(map(float, line[2:].strip().split(',')))
                        i_list.append(values)
                    elif line.startswith("pred: "):
                        values = int(line[6:].strip())
                        i_pred_list.append(values)
                    elif line.startswith("expl:"):
                        # strip off “expl:” and surrounding brackets
                        raw = line[5:].strip().strip('[]')
                        # if raw is non-empty, split & map; otherwise give []
                        values = list(map(int, raw.split(','))) if raw else []
                        expl_list.append(values)
                    elif line.startswith("wit:"):
                        values = list(map(float, line[5:].strip().split(',')))
                        pyxai_wit_list.append(values)
                    elif line.startswith("pred of wit: "):
                        values = int(line[12:].strip())
                        pyxai_pred_wit_list.append(values)

        except FileNotFoundError:
            print(f"File not found: {log_path} — skipping.")
            continue

        # print("i_list:", i_list)
        # print("expl_list:", expl_list)
        tested = set()
        invalid_xp = 0
        valid_xp = 0
        num_wxp = 0
        num_invalid_pyxai_wit = 0
        output_str = ''

        print(base, depth, num)

        for i, (inst, expl) in enumerate(zip(i_list, expl_list)):
            opts.explain = inst
            if tuple(opts.explain) in tested:
                continue
            tested.add(tuple(opts.explain))

            cert = None
            wit = None
            if expl:
                # validating the explanation
                expl = [i-1 for i in expl]
                valid, cert, real_cxp, wit, num_wits, num_proofs = rf_exp.validate(opts.explain, expl, xtype='con', min_max_dict=min_max_dict)
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
            output_str += f"cert: {cert}\n"
            output_str += f"Consistent ML referees: {i_pred_list[i] == rf_exp.f.predict(inst)}\n"
            output_str += f"Valid PyXAI CXp wit: {i_pred_list[i] != pyxai_pred_wit_list[i]}\n"

            if cert is False:
                if wit is None:
                    output_str += "DRAT-TRIM proof failed\n"
                else:
                    output_str += f"RFxpl sub-min wit invalid: {','.join(f'{x}' for x in wit)}\n"
            else:
                output_str += f"CXp: {real_cxp}\n"

                pyxai_wit = pyxai_wit_list[i]
                if rf_exp.f.predict(pyxai_wit) == rf_exp.f.predict(inst):
                    output_str += f"PyXAI wit invalid!\n"
                    num_invalid_pyxai_wit += 1
            output_str += f"Redundant %: {(len(expl) - len(real_cxp)) * 100 / len(expl):.1f}\n\n"

        output_str += f"nof. valid: {valid_xp}\n"
        output_str += f"nof. invalid: {invalid_xp}\n"
        output_str += f"nof. non-minimal: {num_wxp}\n"
        output_str += f"nof. invalid PyXAI's wit: {num_invalid_pyxai_wit}\n"
        open(f"results/pyxai_vs_RFxpl/toy/con/{base}.{num}.log", 'w').write(output_str)
