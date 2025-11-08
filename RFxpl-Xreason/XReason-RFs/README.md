# XReason

Boosted Trees eXplainer as the Checker for validating RFxpl’s and PyXAI’s results.

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

The archive `smt_enc.tar.gz` contains the SMT encodings of the toy Random Forests and the Random Forests with 50 trees used by RFxpl and PyXAI.

Run `python3 check_xp_rfs_smt_enc.py -x abd` to check PyXAI’s AXps on the RF classifiers.
Use the option `-x con` to check PyXAI’s CXPs on the same classifiers.
Run `python3 check_xp_rfs_smt_enc_toy.py -x abd` and `python3 check_xp_rfs_smt_enc_toy.py -x con` to check PyXAI’s AXps and CXPs on the toy RF classifiers.

To check RFxpl’s explanations, run `python3 check_xp_rfxpl.py -x abd` for AXps,  
and run `python3 check_xp_rfxpl.py -x con` for CXps.

The archive `pyxai_results.tar.gz` contains the PyXAI log files required by XReason.  

The archive `results.tar.gz` contains the experimental results of checking both PyXAI’s explanations and RFxpl’s explanations.

To compare the independent checking logs produced by RFxpl and XReason for PyXAI’s results,  
run `compare_logs_axps.py` and `compare_logs_cxps.py`.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
