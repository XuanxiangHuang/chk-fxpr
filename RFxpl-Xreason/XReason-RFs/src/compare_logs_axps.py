#!/usr/bin/env python3
"""
Compare RFxpl and Xreason log files to check agreement on PyXAI explanations.
"""

import os

# ====== EDIT ME ======
DATA_LIST = "datasets.list"
XREASON_BASE = "results/pyxai_vs_xreason/abd"
RFXPL_BASE = "../../RFxpl/results/pyxai_vs_RFxpl/abd"
PATTERN_X = "{base}/{dataset}.{num}.log"
PATTERN_R = "{base}/{dataset}.{num}.log"
NUM = 50
# =====================


def extract_dataset_name(line):
    """
    Extract dataset name from a line that might contain a full path.
    """
    line = line.strip()

    # Remove the last column (space-separated number at the end)
    if ' ' in line:
        line = line.rsplit(' ', 1)[0]  # Remove last space-separated token

    # Extract the filename without extension
    if '/' in line:
        # Get the last part of the path (the filename)
        filename = line.split('/')[-1]
    else:
        filename = line

    # Remove .csv extension if present
    if filename.endswith('.csv'):
        filename = filename[:-4]

    return filename


def parse_log_file(filepath):
    """
    Parse a log file and extract instance blocks.
    Returns a dict mapping instance string to its AXp value.
    """
    instances = {}

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for instance line
        if line.startswith('i:'):
            instance = line[2:].strip()  # Remove 'i:' prefix
            pyxai = None
            axp = None

            # Read the next few lines to find PyXAI and AXp
            j = i + 1
            while j < len(lines) and j < i + 20:  # Look ahead up to 20 lines
                next_line = lines[j].strip()

                if next_line.startswith('i:'):
                    # Found next instance, stop here
                    break
                elif next_line.startswith('PyXAI:'):
                    pyxai = next_line[6:].strip()
                elif next_line.startswith('AXp:'):
                    axp = next_line[4:].strip()

                j += 1

            # Store the instance with its AXp
            if instance and axp is not None:
                instances[instance] = {
                    'pyxai': pyxai,
                    'axp': axp
                }

            i = j  # Jump to the next instance
        else:
            i += 1

    return instances


def compare_logs(xreason_file, rfxpl_file):
    """
    Compare two log files and return statistics.
    """
    xreason_data = parse_log_file(xreason_file)
    rfxpl_data = parse_log_file(rfxpl_file)

    if xreason_data is None or rfxpl_data is None:
        return None

    # Find common instances
    common_instances = set(xreason_data.keys()) & set(rfxpl_data.keys())

    if not common_instances:
        print(f"Warning: No common instances found between files")
        return {
            'total': 0,
            'agree': 0,
            'disagree': 0,
            'xreason_only': len(xreason_data),
            'rfxpl_only': len(rfxpl_data),
            'disagreements': []
        }

    agree = 0
    disagree = 0
    disagreements = []

    for instance in sorted(common_instances):
        xr_axp = xreason_data[instance]['axp']
        rf_axp = rfxpl_data[instance]['axp']

        if xr_axp == rf_axp:
            agree += 1
        else:
            disagree += 1
            disagreements.append({
                'instance': instance,
                'pyxai': xreason_data[instance]['pyxai'],
                'xreason_axp': xr_axp,
                'rfxpl_axp': rf_axp
            })

    return {
        'total': len(common_instances),
        'agree': agree,
        'disagree': disagree,
        'xreason_only': len(xreason_data) - len(common_instances),
        'rfxpl_only': len(rfxpl_data) - len(common_instances),
        'disagreements': disagreements
    }


def main():
    # Read dataset list
    if not os.path.exists(DATA_LIST):
        print(f"Error: Dataset list file '{DATA_LIST}' not found")
        return

    with open(DATA_LIST, 'r') as f:
        datasets = [extract_dataset_name(line) for line in f if line.strip()]

    print("=" * 80)
    print("RFxpl vs Xreason Comparison Report")
    print("=" * 80)
    print()

    overall_agree = 0
    overall_disagree = 0
    overall_total = 0

    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        print("-" * 80)

        xreason_file = PATTERN_X.format(base=XREASON_BASE, dataset=dataset, num=NUM)
        rfxpl_file = PATTERN_R.format(base=RFXPL_BASE, dataset=dataset, num=NUM)

        result = compare_logs(xreason_file, rfxpl_file)

        if result is None:
            print(f"  No data found for this dataset")
            continue

        dataset_total = result['total']
        dataset_agree = result['agree']
        dataset_disagree = result['disagree']

        # Print disagreements for this dataset
        if result['disagreements']:
            print(f"\n  {result['disagree']} disagreements out of {result['total']} instances")
            for d in result['disagreements']:
                print(f"    Instance: {d['instance']}")
                print(f"    PyXAI:    {d['pyxai']}")
                print(f"    Xreason:  {d['xreason_axp']}")
                print(f"    RFxpl:    {d['rfxpl_axp']}")
                print()

        if dataset_total > 0:
            print(f"\nDataset Summary:")
            print(f"  Total instances: {dataset_total}")
            print(f"  Agreements: {dataset_agree} ({100 * dataset_agree / dataset_total:.1f}%)")
            print(f"  Disagreements: {dataset_disagree} ({100 * dataset_disagree / dataset_total:.1f}%)")

            overall_total += dataset_total
            overall_agree += dataset_agree
            overall_disagree += dataset_disagree
        else:
            print(f"  No data found for this dataset")

    print("\n" + "=" * 80)
    print("Overall Summary")
    print("=" * 80)
    if overall_total > 0:
        print(f"Total instances: {overall_total}")
        print(f"Agreements: {overall_agree} ({100 * overall_agree / overall_total:.1f}%)")
        print(f"Disagreements: {overall_disagree} ({100 * overall_disagree / overall_total:.1f}%)")
    else:
        print("No data processed")
    print("=" * 80)


if __name__ == "__main__":
    main()