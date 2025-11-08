# XReason

Boosted Trees eXplainer as the Checker

### Usage

The `toy_bts` folder contains the toy example referenced in the paper.  
Run `python3 check_cxp_toy.py` to reproduce the experiments on this toy example.

The archive `models.tar.gz` contains the Boosted Trees classifiers with depth ≥ 3 and 50 trees.  
To reproduce the corresponding experimental results, run `python3 check_cxp.py`.

The archive `pyxai_results.tar.gz` contains the PyXAI log files required by XReason.  
XReason processes each log file to extract the instances and explanations, and then executes its checking algorithm to validate PyXAI’s CXPs.

The archive `results.tar.gz` contains the experimental results reported in the paper.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
