from __future__ import print_function
from anchor_wrap import anchor_call
from data import Data
from options import Options
import os
import resource
import sys
from xgbooster import XGBooster

if __name__ == '__main__':

    DATASETS = {
        "4_breastw":          {"depth": 3, "num": 50},
        "6_cardio":           {"depth": 3, "num": 50},
        "7_Cardiotocography": {"depth": 3, "num": 50},
        "12_fault":           {"depth": 4, "num": 50},
        "21_Lymphography":    {"depth": 3, "num": 50},
        "29_Pima":            {"depth": 3, "num": 50},
        "30_satellite":       {"depth": 3, "num": 50},
        "31_satimage-2":      {"depth": 3, "num": 50},
        "33_skin":            {"depth": 3, "num": 50},
        "37_Stamps":          {"depth": 3, "num": 50},
        "appendicitis":       {"depth": 3, "num": 50},
        "banknote":           {"depth": 3, "num": 50},
        "biodegradation":     {"depth": 3, "num": 50},
        "glass2":             {"depth": 4, "num": 50},
        "heart-c":            {"depth": 5, "num": 50},
        "ionosphere":         {"depth": 3, "num": 50},
        "magic":              {"depth": 3, "num": 50},
        "mofn-3-7-10":        {"depth": 3, "num": 50},
        "phoneme":            {"depth": 4, "num": 50},
        "ring":               {"depth": 3, "num": 50},
        "sonar":              {"depth": 3, "num": 50},
        "spambase":           {"depth": 4, "num": 50},
        "spectf":             {"depth": 3, "num": 50},
        "twonorm":            {"depth": 3, "num": 50},
        "wdbc":               {"depth": 4, "num": 50},
        "wpbc":               {"depth": 4, "num": 50},
        "xd6":                {"depth": 3, "num": 50},
    }
    

    for base, cfg in DATASETS.items():
        depth = cfg["depth"]
        num   = cfg["num"]
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
