#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## explain.py
##
##  Created on: Aug 1, 2018
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
from __future__ import print_function
import cplex
import numpy as np
import os
from pysat.examples.hitman import Hitman
from pysat.formula import IDPool
from pysmt.shortcuts import Solver
from pysmt.shortcuts import And, BOOL, Implies, Not, Or, Symbol
from pysmt.shortcuts import Equals, GT, Int, Real, REAL
import resource
from six.moves import range
import sys




#
#==============================================================================
class ILPExplainer(object):
    """
        An ILP-inspired minimal explanation extractor for neural networks
        based on ReLUs.
    """

    def __init__(self, oracle, intvs, imaps, ivars, feats, nof_classes,
            options, xgb):
        """
            Constructor.
        """

        self.feats = feats
        self.intvs = intvs
        self.imaps = imaps
        self.ivars = ivars
        self.nofcl = nof_classes
        self.optns = options
        self.idmgr = IDPool()

        # saving XGBooster
        self.xgb = xgb

        self.verbose = self.optns.verb
        self.oracle = oracle

        # input (feature value) variables
        self.inps = self.xgb.extended_feature_names_as_array_strings[:]

        self.outs = []  # output (class  score) variables
        for c in range(self.nofcl):
            self.outs.append('class{0}_score'.format(c))

        # hypotheses
        self.hypos = []

        # adding indicators for correct and wrong outputs
        self.oracle.variables.add(
                names=['csel_{0}'.format(i) for i in range(len(self.outs))],
                types='B' * self.nofcl)
        for i in range(self.nofcl):
            ivar = 'csel_{0}'.format(i)
            wrong = ['wrongc_{0}_{1}'.format(i, j) for j in range(self.nofcl) if i != j]
            self.oracle.variables.add(names=wrong, types='B' * len(wrong))

            # ivar implies at least one wrong class
            self.oracle.indicator_constraints.add(indvar=ivar, lin_expr=[wrong,
                [1] * len(wrong)], sense='G', rhs=1)

            for j in range(self.nofcl):
                if i != j:
                    # iv => (o_j - o_i >= 0.0000001)

                    iv = 'wrongc_{0}_{1}'.format(i, j)
                    ov, oc = [self.outs[j], self.outs[i]], [1, -1]
                    self.oracle.indicator_constraints.add(indvar=iv,
                            lin_expr=[ov, oc], sense='G', rhs=0.0000001)

        # linear constraints activating a specific class
        # will be added for each sample individually
        # e.g. self.oracle.linear_constraints.add(lin_expr=[['c_0'], [1]], senses=['G'], rhs=[1])

    def add_sample(self, sample, expl_ext=None, prefer_ext=False):
        """
            Add constraints for a concrete data sample.
        """

        # transformed sample
        self.sample = list(self.xgb.transform(sample)[0])

        self.hypos = [[] for v in sample]

        # saving external explanation to be minimized further
        if expl_ext == None or prefer_ext:
            self.to_consider = [True for h in self.hypos]
        else:
            eexpl = set(expl_ext)
            self.to_consider = [True if i in eexpl else False for i, h in enumerate(self.hypos)]

        if not self.intvs:
            for i, v in enumerate(self.inps, 1):
                eql = [[v], [1.0]]
                rhs = [float(self.sample[i - 1]) if '_' not in v else int(self.sample[i - 1])]

                cnames = ['hypo_{0}'.format(i)]
                senses = ['E']
                constr = [eql]

                # adding a constraint to the list of hypotheses
                j = int(v.split('_')[0][1:])
                if self.to_consider[j]:
                    assump = self.oracle.linear_constraints.add(lin_expr=constr,
                            senses=senses, rhs=rhs, names=cnames)

                    self.hypos[j].append(tuple([cnames[0], constr, rhs, senses,
                        i - 1, j]))
        else:
            for i, (inp, val) in enumerate(zip(self.inps, self.sample), 1):
                for ub, fvar in zip(self.intvs[inp], self.ivars[inp]):
                    if ub == '+' or val < ub:
                        v = fvar
                        break
                else:
                    assert 0, 'No proper interval found for {0}'.format(feat)

                eql = [[v], [1]]
                rhs = [1]

                cnames = ['hypo_{0}'.format(i)]
                senses = ['E']
                constr = [eql]

                # adding a constraint to the list of hypotheses
                j = int(v.split('_')[0][1:])
                if self.to_consider[j]:
                    assump = self.oracle.linear_constraints.add(lin_expr=constr,
                            senses=senses, rhs=rhs, names=cnames)

                    self.hypos[j].append(tuple([cnames[0], constr, rhs, senses,
                        i - 1, j]))

        # getting the true observation
        # (not the expected one as specified in the dataset)
        self.oracle.solve()
        if self.oracle.solution.is_primal_feasible():
            model = self.oracle.solution
        else:
            assert 0, 'Formula is unsatisfiable under given assumptions'

        # choosing the maximum
        outvals = [float(model.get_values(o)) for o in self.outs]
        maxoval = max(zip(outvals, range(len(outvals))))

        # correct class id (corresponds to the maximum computed)
        self.out_id = maxoval[1]
        self.output = self.xgb.target_name[self.out_id]

        # observation (forcing it to be wrong)
        self.oracle.linear_constraints.add(lin_expr=[[['csel_{0}'.format(self.out_id)], [1]]],
                senses='E', rhs=[1], names=['forced_output'])

        if self.verbose:
            inpvals = self.xgb.readable_sample(sample)

            self.preamble = []
            for f, v in zip(self.xgb.feature_names, inpvals):
                if f not in v:
                    self.preamble.append('{0} = {1}'.format(f, v))
                else:
                    self.preamble.append(v)

            print('  explaining:  "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.output))

    def explain(self, sample, smallest=False, expl_ext=None, prefer_ext=False):
        """
            Hypotheses minimization.
        """

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime

        # add constraints corresponding to the current sample
        self.add_sample(sample)

        # if satisfiable, then the observation is not implied by the hypotheses
        self.oracle.solve()
        if self.oracle.solution.is_primal_feasible():
            print('  no implication!')
            sys.exit(1)

        if not smallest:
            hypos  = [h for h, c in zip(self.hypos, self.to_consider) if not c]
            hypos += [h for h, c in zip(self.hypos, self.to_consider) if c]
            self.hypos = hypos

            #rhypos = self.compute_minimal()
        else:  # get a smallest explanation
            #rhypos = self.compute_smallest()
            pass

        if self.optns.xtype == 'abd':
            rhypos = self.compute_minimal()
        else:
            rhypos = self.compute_cxp()

        self.time = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + \
                resource.getrusage(resource.RUSAGE_SELF).ru_utime - self.time

        # explanation
        expl = [h[0] for h in rhypos]

        if self.verbose:
            self.preamble = [self.preamble[i] for i in expl]
            print('  explanation: "IF {0} THEN {1}"'.format(' AND '.join(self.preamble), self.xgb.target_name[self.out_id]))
            print('  # hypos left:', len(rhypos))
            print('  time: {0:.2f}'.format(self.time))

        # removing hypotheses related to the current sample
        for hypo in rhypos:
            for h in hypo[1]:
                self.oracle.linear_constraints.delete(h[0])

        # removing the output forced to be wrong
        self.oracle.linear_constraints.delete('forced_output')

        return expl

    def compute_minimal(self):
        """
            Compute any subset-minimal explanation.
        """

        # result
        rhypos = []

        # simple deletion-based linear search
        for hypo in self.hypos:
            for h in hypo:
                self.oracle.linear_constraints.delete(h[0])

            self.oracle.solve()
            if self.oracle.solution.is_primal_feasible():
                # this hypothesis is needed
                # adding it back to the list
                for h in hypo:
                    self.oracle.linear_constraints.add(lin_expr=h[1],
                            senses=h[3], rhs=h[2], names=[h[0]])

                rhypos.append(tuple([hypo[0][5], hypo]))

        return rhypos

    def compute_cxp(self):
        """
        Compute a CXp
        """
        rhypos = []

        for hypo in self.hypos:
            for h in hypo:
                self.oracle.linear_constraints.delete(h[0])

        for hypo in self.hypos:
            for h in hypo:
                self.oracle.linear_constraints.add(lin_expr=h[1],
                                                   senses=h[3], rhs=h[2], names=[h[0]])

            self.oracle.solve()
            if not self.oracle.solution.is_primal_feasible():
                for h in hypo:
                    self.oracle.linear_constraints.delete(h[0])
                rhypos.append(tuple([hypo[0][5], hypo]))

        return rhypos
