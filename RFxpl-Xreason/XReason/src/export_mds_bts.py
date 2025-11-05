
from __future__ import print_function
import os
import random
from xgbooster import XGBooster
from options import Options
import pickle

random.seed(1234)

if __name__ == '__main__':
    data_list = "test_mds.list"

    num = 2
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
        # pickle_path = f'../models/{base}/{base}_nbestim_{num}_maxdepth_{depth}_testsplit_0.2.mod.pkl'
        pickle_path = f'toy_bts/{base}/{base}_nbestim_{num}_maxdepth_{depth}_testsplit_0.2.mod.pkl'

        # build an export path that includes the file name (without extension)
        out_base = os.path.join(
            '../../exported_bts',
            f'{base}/{base}_nbestim_{num}_maxdepth_{depth}_testsplit_0.2'
        )

        opts = Options(None)
        opts.n_estimators = num
        opts.files = pickle_path

        xgb = XGBooster(opts, from_model=opts.files)

        xgb.save_model_for_compatibility(
            pickle_model_path=pickle_path,
            out_base=out_base
        )
        print('done')
