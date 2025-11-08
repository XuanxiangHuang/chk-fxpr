# RFxpl

SAT-based Random Forests eXplainer as the Checker

### Usage

The `Classifiers` folder contains the two toy examples referenced in the paper.  
Run `python3 check_cxp_toy.py` and `python3 check_axp_toy.py` to reproduce the experiments on these toy examples.

The archive `Classifiers-50.tar.gz` contains the Random Forest classifiers with depth 4 and 50 trees.  
To reproduce the corresponding experimental results, run `python3 check_cxp.py` and `python3 check_axp.py`, respectively.

The archive `pyxai_results.tar.gz` contains the PyXAI log files required by RFxpl.  
RFxpl processes each log file to extract the instances and explanations, and then executes its checking algorithm to validate PyXAI’s explanations (both AXps and CXPs).

The archive `results.tar.gz` contains the experimental results reported in the paper.
