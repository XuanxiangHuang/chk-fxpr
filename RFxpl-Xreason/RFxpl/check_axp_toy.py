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

    BASE_DIR = "../bench"

    DATASETS = {
        "29_Pima": {"depth": 4, "num": 3},
        "xd6":     {"depth": 4, "num": 4},
    }

    for base, cfg in DATASETS.items():
        depth = cfg["depth"]
        num   = cfg["num"]
        # Load .csv from ../bench/<base>/<base>.csv
        csv_path = os.path.join(BASE_DIR, base, f"{base}.csv")
        df = pd.read_csv(csv_path)
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

        log_path = f"pyxai_results/toy/abd/{base}.{num}.log"

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

        except FileNotFoundError:
            print(f"File not found: {log_path} — skipping.")
            continue

        # print("i_list:", i_list)
        # print("expl_list:", expl_list)
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

            cert = None
            wit = None
            if expl:
                # validating the explanation
                expl = [i-1 for i in expl]
                valid, cert, real_axp, wit, num_wits, num_proofs = rf_exp.validate(opts.explain, expl, xtype='abd', min_max_dict=min_max_dict)
                if valid:
                    valid_xp += 1
                elif len(real_axp) == 0:
                    invalid_xp += 1
                    inst = np.array(opts.explain)
                    real_axp = []
                else:
                    num_wxp += 1
            else:
                real_axp = []
                invalid_xp += 1

            output_str += f"i: {','.join(f'{x}' for x in inst)}\n"
            output_str += f"PyXAI: {expl}\n"
            output_str += f"cert: {cert}\n"
            output_str += f"Consistent ML referees: {i_pred_list[i] == rf_exp.f.predict(inst)}\n"

            if cert is False:
                if wit is not None:
                    output_str += f"RFxpl sub-min wit invalid: {','.join(f'{x}' for x in wit)}\n"
                else:
                    output_str += f"DRAT-TRIM proof failed\n"
            else:
                if len(real_axp):
                    output_str += f"AXp: {real_axp}\n"
                else:
                    if rf_exp.f.predict(wit) == rf_exp.f.predict(inst):
                        output_str += f"RFxpl wit invalid!\n"
                    output_str += f"RFxpl wit: {','.join(f'{x}' for x in wit)}\n"
            output_str += f"Redundant %: {(len(expl) - len(real_axp)) * 100 / len(expl):.1f}\n\n"

        output_str += f"nof. valid: {valid_xp}\n"
        output_str += f"nof. invalid: {invalid_xp}\n"
        output_str += f"nof. non-minimal: {num_wxp}\n"
        open(f"results/pyxai_vs_RFxpl/toy/abd/{base}.{num}.log", 'w').write(output_str)
