# -*- coding: utf-8 -*-
import pymc3 as pm
import numpy as np
import graphviz
import copy
import matplotlib
import daft
import scipy.stats as stats
import re
import theano
import theano.tensor as T
from theano import pp
from scipy import optimize


def model_info(model):
    print 'Model RVs', model.named_vars.values()
    for k, v in model.named_vars.iteritems():
        print v,
        print v.type,
        if k in model.test_point:
            print model.test_point[k].shape,
        if v in model.disc_vars:
            print 'discrete',
        if v in model.cont_vars:
            print 'continuous',
        if v in model.free_RVs:
            print 'free',
        if v in model.observed_RVs:
            print 'observed',
        if v in model.potentials:
            print 'potential',
        if v in model.deterministics:
            print 'deterministic',
            print ''
            print 'Model test point logp', model.logp(model.test_point)
            # FIXME This is annoyingly slow.. maybe we could somehow stop bfgs earlier?
            # print 'Model max bfgs logp', model.logp(pm.find_MAP(model=model))
    print 'Model max powell logp', model.logp(pm.find_MAP(model=model, fmin=optimize.fmin_powell))


def model_profile(m):
    m.profile(m.logpt).summary()


def burn_and_thin(trace, burn, thin):
    return trace[burn::thin]


def model_sample(m, nr, burn=1, thin=1, **kwargs):
    return burn_and_thin(pm.sample(nr, **kwargs), burn, thin)


def model_sample_nuts(m, nr, sample_args={}, nuts_args={}):
    with m:
        return model_sample(m, nr, step=pm.NUTS(**nuts_args), **sample_args)


def model_sample_metropolis(m, nr, sample_args={}, step_args={}):
    with m:
        return model_sample(m, nr, step=pm.Metropolis(**step_args), **sample_args)


def model_sample_bfgs_nuts(m, nr, sample_args={}, nuts_args={}):
    with m:
        start = pm.find_MAP()
        return model_sample(m, nr, start=start, step=pm.NUTS(**nuts_args), **sample_args)


def model_sample_advi_nuts(m, nr, sample_args={}, nuts_args={}, advi_args={}):
    with m:
        if 'n' not in advi_args:
            advi_args['n'] = 10000
        mu, sds, elbo = pm.variational.advi(**advi_args)
        return model_sample(m, nr, pm.NUTS(scaling=m.dict_to_array(sds),
                                           is_cov=True, **nuts_args),
                            start=mu, **sample_args)


def model_sample_advi(m, nr, sample_args={}, advi_args={}):
    with m:
        if 'n' not in advi_args:
            advi_args['n'] = nr
        fit = pm.variational.advi(**advi_args)
        burn = sample_args.pop('burn', 1)
        thin = sample_args.pop('thin', 1)
        return burn_and_thin(pm.variational.sample_vp(fit, nr, **sample_args), burn, thin)
