# XReason

Boosted Trees eXplainer as the Checker

### Installation

The following packages are necessary to run XReason:

* [Anchor](https://github.com/marcotcr/anchor) (version [0.0.2.0](https://pypi.org/project/anchor-exp/0.0.2.0/))
* [anytree](https://anytree.readthedocs.io/)
* [LIME](https://github.com/marcotcr/lime)
* [namedlist](https://gitlab.com/ericvsmith/namedlist)
* [numpy](http://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pySMT](https://github.com/pysmt/pysmt) (with Z3 installed)
* [PySAT](https://github.com/pysathq/pysat)
* [scikit-learn](https://scikit-learn.org/stable/)
* [SHAP](https://github.com/slundberg/shap)
* [XGBoost](https://github.com/dmlc/xgboost/) (version [1.7.5](https://pypi.org/project/xgboost/1.7.5/))

**Important:** If you are using a MacOS system, please make sure you use `libomp` (OpenMP) version 11. Later versions are affected by [this bug](https://github.com/dmlc/xgboost/issues/7039).

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
