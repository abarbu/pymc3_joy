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
from pymc3_extra import *
from collections import defaultdict, namedtuple
from tabulate import tabulate

# How to use me from a notebook util this becomes a real package.
#
# git checkout git@github.com:abarbu/pymc3_joy.git
#
# Add this to your file:
#
# import sys
# sys.path.append(os.getcwd()+'/pymc3_joy/')
# from pymc3_extra import *
# from pymc3_wrap import *
#
# If you want to hack on this file in the notebook sprinkle:
# reload(pymc3_wrap)
# reload(pymc3_extra)
# from pymc3_wrap import *
# from pymc3_extra import *
# in your code, edit the file, and rerun the cell.

# fix, printing, and m_eval are very handy for debugging. Fix whatever
# values you want, print out others, and evaluate the model to see its
# likelihood. m_eval_random is also handy to pick some random point.

# Wrap up as a monad. m_merge is >>. const is return. fix is also a
# form of return since it replaces a node with a constant.
#
# model_ functions operate on pymc3 models.
# m_ functions operate on wrapped models, ie. those of class M.

# FIXME What's going on with observations? How do we observe two nodes
# jointly?
# FIXME Check shapes to avoid strange errors
# FIXME marginalize values
# FIXME show shape in the nodes
# FIXME Time series models
# FIXME The default testvals are shitty for hierarchical models. Randomize them.
# FIXME minibatch ADVI
# FIXME pm.df_summary(trace)
# FIXME To wrap up:
# https://github.com/pymc-devs/pymc3/blob/master/docs/source/notebooks/cox_model.ipynb
#  _ = pm.traceplot(traces_signoise[-1000:], figsize=(12,len(traces_signoise.varnames)*1.5),
#            lines={k: v['mean'] for k, v in pm.df_summary(traces_signoise[-1000:]).iterrows()})
# https://pymc-devs.github.io/pymc3/notebooks/posterior_predictive.html

# Runtime debugging:
# FIXME Draw 1 sample and 1 good sample
# FIXME Treat the model as a function and show intermediate values
# FIXME Call function interface

# FIXME Plots should be able to only show outputs/inputs
# FIXME Posterior predictive stuff
# FIXME NN Interface for ADVI

# FIXME ADVI has a bug with step sizes when computing the next
# value. If the gradients are extremely (but not near underflowing) it
# eventually attempts to set the next value to nan. This can be seen
# easily with the logistic regression if it's not clipped.

# Other fun things
# https://stackoverflow.com/questions/24242660/pymc3-multiple-observed-values
# https://stackoverflow.com/questions/30798447/porting-pymc2-code-to-pymc3-hierarchical-model-for-sports-analytics
# https://stackoverflow.com/questions/22708513/porting-pymc2-bayesian-a-b-testing-example-to-pymc3
# https://stackoverflow.com/questions/27382474/pymc3-parallel-computing-with-njobs1-vs-gpu
# https://github.com/junpenglao/Bayesian-Cognitive-Modeling-in-Pymc3
# https://github.com/nwngeek212/ProbabilisticProgramming/blob/master/ProbabilisticProgramming.ipynb
# https://github.com/aloctavodia/Statistical-Rethinking-with-Python-and-PyMC
# https://github.com/dadaromeo/bayes-dap
# http://pymc-devs.github.io/pymc3/notebooks/bayesian_neural_network_advi.html
# https://blog.quantopian.com/bayesian-deep-learning2/

# Error guide:
#
# "has no finite default value to use," it's likely you are doing
# something like uniform(a,a). pymc3 doesn't like this because it has
# no values.
#
# "has no attribute 'name'" you are passing in a non-Theano object to
# a context that expects one.
#
# "floating point exception" this is usually a crash. You're putting
# an integer in a float context or a float in an integer context in
# Theano. (FIXME Check pymcdebug.fix(a,T.constant(0.9)). Types of T
# should match with that type of a.)
#
# NUTS is slow
#
#  Your model is very hard to sample from because there are no
#  reasonable samples to draw. Perhaps the priors are bad.
#
#  Your model is underdetermined. NUTS is slow for models which have
#  very little data with many free parameters. This basically means
#  that NUTS gets the right answer quickly and keeps looking for
#  something better but can't find it because the gradients don't
#  help.

# Some ideas behind this API:
#
# There are several problems with pymc models. Easy to miss
# variables. Hard to keep track of samples. Want an acceptance ratio
# or something like it. Want to know if our solutions are good. Want
# to package up models more easily.
#
# So we need some basic model info.
# How do we compute the determinsitic variables as well?
# We should specify the model in some way so we can run it easily
# Like, provide input data easily and sample from the model.
#
# We want some sort of set of model combinators then.
#
# First we want to be able to run the model forwards.
# Then to add noise to any part of the model.
#
# Want some way to print the model and its structure in something like
# plate notation.

M_DETERMINISTIC_OBSERVATION_SD = 0.01

class M():
    def __init__(self, vs, output):
        # a list of variables in this model. The order matters they will be emitted in order.
        self.vs = vs
        # models have only a single output like any function. This keeps everything simple.
        self.output = output
    def __str__(self):
        s = []
        s.append('Model RVs {}'.format(m_names(self)))
        def fone(v, name):
            s.append(' {} {}[{}] {} observed={} fixed={}'.format(name, v.var_type, v.shape, v.pymc3fn, bool1(v.observed), bool1(v.fixed)))
        def fedge(location, vin, namein, valin, v, name, val):
            return ()
        def fargs(v, name, args, kwargs):
            return ()
        m_walk(fone, fedge, fargs, self).keys()
        return '\n'.join(s)

# NOTE No function should ever return a naked V(). They should always be wrapped in M().
class V():
    def __init__(self, pymc3fn, args, kwargs, shape, var_type, var_kind, node_type, maybe_name, observed, fixed, printing):
        # var_type is continuous, discrete, deterministic, potential, etc.
        # var_kind is unif, N, Dir, Beta, etc.
        # node_type is either a theano TensorVariable, None, or 'unknown'
        self.pymc3fn = pymc3fn
        self.args = args
        self.kwargs = kwargs
        self.shape = shape
        self.var_type = var_type
        self.var_kind = var_kind
        self.node_type = node_type
        self.maybe_name = maybe_name
        self.observed = observed
        self.fixed = fixed
        self.printing = printing
        if self.observed and self.fixed:
            raise ValueError("Can't be both fixed and observed")

# Misc functions (should likely go elsewhere)

def is_subset(x, y): # is x a subset of y
    for i in x:
        if i not in y:
            return False
    return True

def bool1(x): # TODO Change observed and fixed so we don't need this anymore...
    return not (x is False or x is None)

def deduplicate(lst):  # keeps order
    uniq = []
    for i in lst:
        if i not in uniq:
            uniq.append(i)
    return uniq

def replace_TeX(s, n):
    return re.sub("\\$.*?\\$", 'm'*n, s)

def strip_locations(l):
    r = []
    for i in l:
        r.append(i[1])
    return deduplicate(r)

def xs(d):
    r = []
    for k, o in d.iteritems():
        r.append(o[0])
    return r

def ys(d):
    r = []
    for k, o in d.iteritems():
        r.append(o[1])
    return r

# Internal utilies

# Create a version of this model that outputs a different
# variable. Not part of the API as Vs are not externally visible.
def m_output(m, v):
    assert v in m.vs # This would be crazy..
    r = copy.copy(m)
    r.output = v
    return r

def m_gensym(m, other_names, i, name):
    # Don't rely on this being deterministic
    nth = 0
    total = 0
    for j,n in enumerate(other_names):
        if n == name:
            total += 1
            if j < i:
                nth += 1
    if name:
        if total == 1:
            return('%s' % name)
        else:
            return('%s_%d' % (name, nth))
    else:
        return('_%d' % nth)

def m_assigned_names(m):
    return map(lambda x: x.maybe_name, m.vs)

# The output is that of m2; this is >>
def m_merge(m1, m2):
    return M(deduplicate(m1.vs + m2.vs), m2.output)

def ms_merge(ms):
    return reduce(m_merge, ms)

def extract_dependencies_args(args):
    r = []
    for k, v in enumerate(args):
        if isinstance(v, M):
            r.append((k, v))
    return r

def extract_dependencies_kwargs(kwargs):
    r = []
    for k, v in kwargs.iteritems():
        if isinstance(v, M):
            r.append((k, v))
    return r

def dependencies_ms(args, kwargs):
    return deduplicate(strip_locations(extract_dependencies_args(args) + extract_dependencies_kwargs(kwargs)))

def dependencies_vs(args, kwargs):
    return map(lambda x: x.output, deduplicate(strip_locations(extract_dependencies_args(args) + extract_dependencies_kwargs(kwargs))))

def dependencies_vs_location(args, kwargs):
    return map(lambda x: (x[0], x[1].output), extract_dependencies_args(args) + extract_dependencies_kwargs(kwargs))

def v_dependencies(v):
    return dependencies_vs(v.args, v.kwargs)

def v_dependencies_location(v):
    return dependencies_vs_location(v.args, v.kwargs)

# This is only useful for display and debugging. Never use this in
# code. Names can be random and unstable.
def m_names(m):
    def fone(v, name):
        return ()
    def fedge(location, vin, namein, valin, v, name, val):
        return ()
    def fargs(v, name, args, kwargs):
        return ()
    return m_walk(fone, fedge, fargs, m).keys()

# This is only useful for display and debugging. Never use this in
# code. Names can be random and unstable.
def m_name_map(m):
    def fone(v, name):
        return ()
    def fedge(location, vin, namein, valin, v, name, val):
        return ()
    def fargs(v, name, args, kwargs):
        return ()
    l = m_walk(fone, fedge, fargs, m)
    return dict(zip(map(lambda x: x[0], l.values()), l.keys()))

# This is our workhorse and the only way to manipulate a model as a whole.
def m_walk(fone, fedge, fargs, m):
    # fone :: V -> Name -> Val
    #  for each node
    # fedge :: Location -> V -> Name -> Val -> V -> Name -> Val -> ()
    #  for each dependency
    # fargs :: V -> Name -> Args[Val] -> KWArgs[Val] -> Val
    #  with args/kwargs replaced by values
    #  this replaces the value returned by fone!
    #
    # Returns a map of names to (V, Val) to reconstruct what was
    # generated without the need to have stable names.
    names = {}
    values = {}
    assigned_names = m_assigned_names(m)
    for i, v in enumerate(m.vs):
        names[v] = m_gensym(m, assigned_names, i, v.maybe_name)
        values[v] = fone(v, names[v])
        for j, l in enumerate(v_dependencies_location(v)):
            fedge(l[0], l[1], names[l[1]], values[l[1]], v, names[v], values[v])
        args = list(v.args)
        for i, o in enumerate(args):
            if isinstance(o, M):
                args[i] = values[o.output]
        args = tuple(args)
        kwargs = copy.copy(v.kwargs)
        for k, o in kwargs.iteritems():
            if isinstance(o, M):
                kwargs[k] = values[o.output]
        values[v] = fargs(v, names[v], args, kwargs)
    r = {}
    for v, name in names.iteritems():
        r[name] = (v, values[v])
    return r

# Conver to a pymc model
def m_pymc3(m, verbose=False):
    if verbose:
        print "Generating PyMC3 model"
    with pm.Model() as model:
        # basic info about the model we're outputting
        model_data = []
        def fone(v, name):
            return v
        def fedge(location, vin, namein, valin, v, name, val):
            return ()
        def fargs(v, name, args, kwargs):
            # pymc3 won't allow us to observe deterministic variables
            # or custom densities. There's nothing wrong with this in
            # principle so we wrap them up in a gaussian with a small
            # SD to pretend that we can observe them.
            # See comment in observe()
            #
            # FIXME Is this really the best way to observe a
            # deterministic variable? From a design point of view it
            # would be more elegant to use a potential so that it
            # doesn't appear in the graph.
            if v.observed and v.var_type == 'deterministic':
                kwargs = copy.copy(kwargs)
                observed = kwargs.pop('observed')
                model_data.append([name + '!ORIG', v.pymc3fn.__name__, args, kwargs])
                model_data.append([name, pm.Normal.__name__, M_DETERMINISTIC_OBSERVATION_SD, observed])
                r = pm.Normal(name,
                              v.pymc3fn(name + '!ORIG', args, kwargs),
                              sd=M_DETERMINISTIC_OBSERVATION_SD,
                              observed=observed)
            elif bool1(v.fixed):
                # This is ok, we don't need to make it into a node as
                # long as it's a theano object.
                model_data.append([name + '!FIXED', v.fixed, [], {}])
                r = v.fixed
            else:
                model_data.append([name, v.pymc3fn.__name__, args, kwargs])
                r = v.pymc3fn(name, *args, **kwargs)
            if v.printing:
                r = T.printing.Print(name)(r)
            return r
        m_walk(fone, fedge, fargs, m)
        if verbose:
            print tabulate(model_data, headers=['RV', 'Type', 'Args', 'KWArgs'])
            print "Finished generating PyMC3 model"
    return model

# Distributions

def m_wrap(f, var_type, var_kind, node_type, *args, **kwargs):
    #    https://github.com/pymc-devs/pymc3/issues/829
    name = kwargs.pop('name', False)
    v = V(f,
          args,
          kwargs,
          kwargs.get('shape', 1),
          var_type,
          var_kind,
          node_type,
          name,
          'observed' in kwargs,
          False,
          False)
    ms = dependencies_ms(args, kwargs)
    ms.append(M([v], v))
    return ms_merge(ms)

def uniform(*args, **kwargs):
    #([lower, upper, transform])	Continuous uniform log-likelihood.
    return m_wrap(pm.Uniform, 'continuous', '$\\mathcal{U}$', T.scalar('x', dtype='floatX'), *args, **kwargs)

def flat(*args, **kwargs):
    #(*args, **kwargs)	Uninformative log-likelihood that returns 0 regardless of the passed value.
    return m_wrap(pm.Flat, 'continuous', 'flat', T.scalar('x', dtype='floatX'), *args, **kwargs)

def normal(*args, **kwargs):
    #(*args, **kwargs)	Univariate normal log-likelihood.
    return m_wrap(pm.Normal, 'continuous', '$\\mathcal{N}$', T.scalar('x', dtype='floatX'), *args, **kwargs)

def beta(*args, **kwargs):
    #([alpha, beta, mu, sd])	Beta log-likelihood.
    return m_wrap(pm.Beta, 'continuous', 'Beta', T.scalar('x', dtype='floatX'), *args, **kwargs)

def exponential(*args, **kwargs):
    #(lam, *args, **kwargs)	Exponential log-likelihood.
    return m_wrap(pm.Exponential, 'continuous', 'Exp', T.scalar('x', dtype='floatX'), *args, **kwargs)

def laplace(*args, **kwargs):
    #(mu, b, *args, **kwargs)	Laplace log-likelihood.
    return m_wrap(pm.Laplace, 'continuous', 'Lap', T.scalar('x', dtype='floatX'), *args, **kwargs)

def studentT(*args, **kwargs):
    #(nu[, mu, lam, sd])	Non-central Student’s T log-likelihood.
    return m_wrap(pm.StudentT, 'continuous', 't-dist', T.scalar('x', dtype='floatX'), *args, **kwargs)

def cauchy(*args, **kwargs):
    #(alpha, beta, *args, **kwargs)	Cauchy log-likelihood.
    return m_wrap(pm.Cauchy, 'continuous', 'cauchy', T.scalar('x', dtype='floatX'), *args, **kwargs)

def halfCauchy(*args, **kwargs):
    #(beta, *args, **kwargs)	Half-Cauchy log-likelihood.
    return m_wrap(pm.HalfCauchy, 'continuous', '1/2-cauchy', T.scalar('x', dtype='floatX'), *args, **kwargs)

def gamma(*args, **kwargs):
    #([alpha, beta, mu, sd])	Gamma log-likelihood.
    return m_wrap(pm.Gamma, 'continuous', 'Gamma', T.scalar('x', dtype='floatX'), *args, **kwargs)

def weibull(*args, **kwargs):
    #(alpha, beta, *args, **kwargs)	Weibull log-likelihood.
    return m_wrap(pm.Weibull, 'continuous', 'Weibull', T.scalar('x', dtype='floatX'), *args, **kwargs)

def studentTpos(*args, **kwargs):
    #(*args, **kwargs)
    return m_wrap(pm.StudentTpos, 'continuous', 't-dist+', T.scalar('x', dtype='floatX'), *args, **kwargs)

def logNormal(*args, **kwargs):
    #([mu, sd, tau])	Log-normal log-likelihood.
    return m_wrap(pm.Lognormal, 'continuous', 'log$\\mathcal{N}$', T.scalar('x', dtype='floatX'), *args, **kwargs)

def chiSquared(*args, **kwargs):
    #(nu, *args, **kwargs)	χ2χ2 log-likelihood.
    return m_wrap(pm.ChiSquared, 'continuous', '$\\chi^{2}$', T.scalar('x', dtype='floatX'), *args, **kwargs)

def halfNormal(*args, **kwargs):
    #([sd, tau])	Half-normal log-likelihood.
    return m_wrap(pm.HalfNormal, '$\\mathcal{N}^{+}$', T.scalar('x', dtype='floatX'), *args, **kwargs)

def wald(*args, **kwargs):
    #([mu, lam, phi, alpha])	Wald log-likelihood.
    return m_wrap(pm.Wald, 'continuous', 'wald', T.scalar('x', dtype='floatX'), *args, **kwargs)

def pareto(*args, **kwargs):
    #(alpha, m, *args, **kwargs)	Pareto log-likelihood.
    return m_wrap(pm.Pareto, 'continuous', 'pareto', T.scalar('x', dtype='floatX'), *args, **kwargs)

def inverseGamma(*args, **kwargs):
    #(alpha[, beta])	Inverse gamma log-likelihood, the reciprocal of the gamma distribution.
    return m_wrap(pm.InverseGamma, 'continuous', '$\\text{Gamma}^-1$', T.scalar('x', dtype='floatX'), *args, **kwargs)

def exGaussian(*args, **kwargs):
    #(mu, sigma, nu, *args, **kwargs)	Exponentially modified Gaussian log-likelihood.
    return m_wrap(pm.ExGaussian, 'continuous', 'Exp(Gamma)', T.scalar('x', dtype='floatX'), *args, **kwargs)

def binomial(*args, **kwargs):
    #(n, p, *args, **kwargs)	Binomial log-likelihood.
    return m_wrap(pm.Binomial, 'discrete', 'Binomial', T.scalar('x', dtype='int32'), *args, **kwargs)

def betaBinomial(*args, **kwargs):
    #(alpha, beta, n, *args, **kwargs)	Beta-binomial log-likelihood.
    return m_wrap(pm.BetaBinomial, 'discrete', 'BetaBin', T.scalar('x', dtype='int32'), *args, **kwargs)

def bernoulli(*args, **kwargs):
    #(p, *args, **kwargs)	Bernoulli log-likelihood
    return m_wrap(pm.Bernoulli, 'discrete', 'Bern', T.scalar('x', dtype='int32'), *args, **kwargs)

def poisson(*args, **kwargs):
    #(mu, *args, **kwargs)	Poisson log-likelihood.
    return m_wrap(pm.Poisson, 'discrete', 'Poisson', T.scalar('x', dtype='int32'), *args, **kwargs)

def negativeBinomial(*args, **kwargs):
    #(mu, alpha, *args, **kwargs)	Negative binomial log-likelihood.
    return m_wrap(pm.NegativeBinomial, 'discrete', 'NegBin', T.scalar('x', dtype='int32'), *args, **kwargs)

def constantDist(*args, **kwargs):
    #(c, *args, **kwargs)	Constant log-likelihood.
    return m_wrap(pm.ConstantDist, 'discrete', 'Const', T.scalar('x', dtype='int32'), *args, **kwargs)

def zeroInflatedPoisson(*args, **kwargs):
    #(theta, psi, *args, **kwargs)	Zero-inflated Poisson log-likelihood.
    return m_wrap(pm.ZeroInflatedPoisson, 'discrete', 'ZIPos', T.scalar('x', dtype='int32'), *args, **kwargs)

def discreteUniform(*args, **kwargs):
    #(lower, upper, *args, **kwargs)	Discrete uniform distribution.
    return m_wrap(pm.DiscreteUniform, 'discrete', 'disc', T.scalar('x', dtype='int32'), *args, **kwargs)

def geometric(*args, **kwargs):
    #(p, *args, **kwargs)	Geometric log-likelihood.
    return m_wrap(pm.Geometric, 'discrete', 'Geom', T.scalar('x', dtype='int32'), *args, **kwargs)

def categorical(*args, **kwargs):
    #(p, *args, **kwargs)	Categorical log-likelihood.
    return m_wrap(pm.Categorical, 'discrete', 'cat', T.scalar('x', dtype='int32'), *args, **kwargs)

def mvNormal(*args, **kwargs):
    #(mu[, cov, tau])	Multivariate normal log-likelihood.
    return m_wrap(pm.MvNormal, 'continuous', 'mv$\\mathcal{N}$', T.vector('x', dtype='floatX'), *args, **kwargs)

def wishart(*args, **kwargs):
    #(n, V, *args, **kwargs)	Wishart log-likelihood.
    return m_wrap(pm.Wishart, 'continuous', 'Wishart', T.vector('x', dtype='floatX'), *args, **kwargs)

def lkjCorr(*args, **kwargs):
    #(n, p, *args, **kwargs)	The LKJ (Lewandowski, Kurowicka and Joe) log-likelihood.
    return m_wrap(pm.LKJCorr, 'continuous', 'LKJ', T.matrix('x', dtype='floatX'), *args, **kwargs)

def multinomial(*args, **kwargs):
    #(n, p, *args, **kwargs)	Multinomial log-likelihood.
    return m_wrap(pm.Multinomial, 'discrete', 'multi', T.vector('x', dtype='int32'), *args, **kwargs)

def dirichlet(*args, **kwargs):
    #(a[, transform])	Dirichlet log-likelihood.
    return m_wrap(pm.Dirichlet, 'discrete', 'Dir', T.vector('x', dtype='int32'), *args, **kwargs)

def flip(*args, **kwargs):
    return bernoulli(*args, **kwargs)

# Special nodes:

# We don't expose pm.Deterministic only fn.
# Instead of pm.Deterministic("A", a-b)
# You want fn(-, a, b)
def fn(fun, *args, **kwargs):
    def f(name, *args, **kwargs):
        return pm.Deterministic(name, fun(*args), **kwargs)
    f.__name__ = 'fn(' + fun.__name__ + ')'
    return m_wrap(f, 'deterministic', 'fn', 'unknown', *args, **kwargs)

def const(val, **kwargs):
    x = normal(1, 0.001)
    fix(x, val)
    return x

def density(fun, *args, **kwargs):
    def f(name, *args, **kwargs):
        return pm.DensityDist(name, fun, *args, **kwargs)
    # TODO Is this output type always correct?
    return m_wrap(f, 'density', 'density', T.scalar('x', dtype='floatX'), *args, **kwargs)

# Unlike other functions this one injects a into a model rather than
# compose with one.
# FIXME This is rather inelegant. It's because models return values but potentials don't.
def add_potential(m, fn, *args, **kwargs):
    def f(name, *args, **kwargs):
        return pm.Potential(name, fn, *args, **kwargs)
    name = kwargs.get('name', False)
    kwargs.pop('name', None)
    m.vs.append(V(pm.Potential,
                  args,
                  kwargs,
                  kwargs.get('shape', 1),
                  'potential',
                  '$\\psi$',
                  None,
                  name,
                  'observed' in kwargs,
                  False,
                  False))
    return m

# Manipulate nodes after the fact

def printing(m, toggle=True):
    m.output.printing = True
    return m

def compatible_data(m, data):
    # TODO Check shape compatibility
    # TODO Check dtype compatibility
    if isinstance(data, theano.Variable):
        return data
    elif isinstance(data, np.ndarray):
        return theano.shared(data)
    else:
        # NOTE This works for scalars as well!
        return theano.shared(np.asarray(data))

# Note that we allow observing deterministic variables even if pymc3
# forbids it. See comment in m_pymc3 about how we do this. But this
# won't work well with certain samplers that make single-variable
# moves.
def observe(m, data):
    # We rely on the fact that Vs are shared between models and _never_ rebuilt or copied
    #
    # FIXME, this is not really a good idea. Much better would be to
    # somehow identify this var and update our own copy of it while
    # leaving the original intact.
    if m.output.fixed:
        raise ValueError("You can't observed a fixed variable. This doesn't make sense")
    m.output.kwargs['observed'] = compatible_data(m, data)
    m.output.observed = True
    return m

def unobserve(m):
    m.output.kwargs.pop('observed', None)
    m.output.observed = False
    return m

def fix(m, data):
    # We rely on the fact that Vs are shared between models and _never_ rebuilt or copied
    #
    # pymc3 doesn't have the notion of a fixed variable. So instead we
    # swap out the distribution for a constant deterministic variable.
    if m.output.observed:
        raise ValueError("You can't fix an observed variable. This doesn't make sense")
    m.output.fixed = compatible_data(m, data)
    return m

def unfix(m, data):
    m.output.fixed = False
    return m

# Visualizing models

def graphviz_locations(g):
    # returns the node locations in this graph; name->(x,y)
    r = {}
    for o in filter(lambda x: x[0] == 'node', map(lambda x: x.split(), g.pipe().splitlines())):
        r[o[1].strip('"')] = (float(o[2]), float(o[3]), float(o[4])/float(o[5]))
    return r

def m_layout(m, flabel=None):
    # use graphviz to lay out each node
    # returns a map from V to (x,y)
    g = graphviz.Digraph(format='plain')
    def fone(v, name):
        label = name
        if flabel:
            label = flabel(v, name)
            # TeX expressions aren't rendered by dot so we replace them
            # with a fixed but small size otherwise they'll take up too
            # much space. Replace them all by that many m's
        g.node(name, label=replace_TeX(label, 2))
    def fedge(location, vin, namein, valin, v, name, val):
        g.edge(namein, name)
    def fargs(v, name, args, kwargs):
        return v
    w = m_walk(fone, fedge, fargs, m)
    l = graphviz_locations(g)
    r = {}
    for name, o in w.iteritems():
        r[o[0]] = l[name]
    return r

def m_daft_pgm(m, flabel=None):
    layout = m_layout(m, flabel)
    pgm = daft.PGM(shape=[max(xs(layout)) + 1, max(ys(layout))+1],
                   origin=[-0.1, -0.1],
                   grid_unit=2,
                   node_unit=1)
    def fone(v, name):
        label = name
        if flabel:
            label = flabel(v, name)
        pgm.add_node(daft.Node(name, label, layout[v][0], layout[v][1], observed=bool1(v.observed), fixed=bool1(v.fixed), aspect=layout[v][2]))
    def fedge(location, vin, namein, valin, v, name, val):
        pgm.add_edge(namein, name)
    def fargs(v, name, args, kwargs):
        return ()
    m_walk(fone, fedge, fargs, m)
    return pgm

def m_draw(m, flabel=None):
    return m_daft_pgm(m, flabel).render()

def m_draw_info(m):
    return m_draw(m, flabel=lambda v, n: n + ' ~ ' + v.var_kind)
 
# Visualization of results

def m_graph_fn(fn, trace, overall_m, ms=None, **kwargs):
    if isinstance(ms, M):
        ms = [ms]
    if isinstance(ms, list):
        d = m_name_map(overall_m)
        ms = map(lambda m: d[m.output], ms)
    fn(trace, varnames=ms, **kwargs)

def m_traceplot(trace, overall_m, ms=None, **kwargs):
    m_graph_fn(pm.traceplot, trace, overall_m, ms, **kwargs)

def m_forestplot(trace, overall_m, ms=None, **kwargs):
    m_graph_fn(pm.forestplot, trace, overall_m, ms, **kwargs)

def m_plot_posterior(trace, overall_m, ms=None, **kwargs):
    m_graph_fn(pm.plot_posterior, trace, overall_m, ms, **kwargs)

def m_autocorrplot(trace, overall_m, ms=None, **kwargs):
    m_graph_fn(pm.autocorrplot, trace, overall_m, ms, **kwargs)

def m_trace_variable(trace, overall_m, m):
    return trace[m_name_map(overall_m)[m.output]]

# Runtime debugging

def m_profile(m):
    return model_profile(m_pymc3(m))

# The digraph of the computation as adjacency maps; a map from V to a
# list of variables. The first map gives the inbound nodes (connected
# by edges going to the node) to the current node, and the second
# gives the outbound nodes from the current node. This is redundant
# but handy.
def m_node_adjacencies(m):
    r_in = defaultdict(list)
    r_out = defaultdict(list)
    def fone(v, name):
        return v
    def fedge(location, vin, namein, valin, v, name, val):
        r_in[v].append(vin)
        r_out[vin].append(v)
    def fargs(v, name, args, kwargs):
        return ()
    m_walk(fone, fedge, fargs, m)
    return r_in, r_out

# Returns the in/out-degree of each node in the graph as a map to in/out tuples.
Degree = namedtuple('Degree', ['in_', 'out_'])
def m_node_degrees(m):
    ins, outs = m_node_adjacencies(m)
    r = {}
    for v in m.vs:
        r[v] = Degree(len(ins[v]), len(outs[v]))
    return r

# If you think of a model as a deterministic function, free random
# variables, are inputs.
def m_inputs(m):
    d = m_node_degrees(m)
    return map(lambda x: m_output(m, x), filter(lambda v: d[v].in_ == 0, m.vs))

# These are all the random variables that have no outgoing edges. If
# you think of the model as a deterministic function these would be
# the outputs.
def m_outputs(m):
    d = m_node_degrees(m)
    return map(lambda x: m_output(m, x), filter(lambda v: d[v].out_ == 0, m.vs))

def m_eval(m, vs_map, verbose=False):
    if not is_subset(map(lambda x: x.output, vs_map.keys()), map(lambda x: x.output, m_inputs(m))):
        raise ValueError("Variables that aren't inputs")
    if verbose:
        for v in m.vs:
            v.printing = True
    for k, v in vs_map.iteritems():
        fix(k, v)
    model = m_pymc3(m)
    like = model.logp(model.test_point)
    if verbose:
        for v in m.vs:
            v.printing = False
    return like

def m_eval_random(m, vs_map, warmup_nr=10, verbose=False):
    if not is_subset(map(lambda x: x.output, vs_map.keys()), map(lambda x: x.output, m_inputs(m))):
        raise ValueError("Variables that aren't inputs")
    # This warms pymc up and takes a few steps away from the default
    # values. We take the steps with a metropolis sampler since we
    # don't care about the results, we just want them to be random.
    model = m_pymc3(m)
    model_info(model)
    warmup_trace = model_sample_metropolis(model, warmup_nr)
    if verbose:
        for v in m.vs:
            v.printing = True
        model = m_pymc3(m)
    trace = model_sample_metropolis(model, 2, sample_args={'trace': warmup_trace})
    if verbose:
        for v in m.vs:
            v.printing = False
    return trace[-1], model.logp(trace[-1])

# Tests

def polynomial_fn(cs, xs):
    components, updates = theano.scan(fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
                                      outputs_info=None,
                                      sequences=[cs, theano.tensor.arange(100)],
                                      non_sequences=xs)
    return components.sum(axis=0)

def test_polynomial():
    cs = normal(2, sd=10, shape=4, name='$c$')
    xs = normal(-2, sd=10, name='$x$', testval=0.1)
    exact_ys = fn(polynomial_fn, cs, xs, name='f(x)')
    # the 'correct' way to do this is to generate a normal noise and
    # add it to exact_ys but pymc won't let us observe a deterministic
    # variable for some reason. Our API allows this.
    ys = normal(exact_ys, sd=0.1, name='$\\epsilon$', testval=0.1)
    return ys

def test_polynomial_right_way():
    # The right way to write a polynomial.
    cs = normal(2, sd=10, shape=4, name='$c$')
    xs = normal(-2, sd=10, name='$x$', testval=0.1)
    exact_ys = fn(polynomial_fn, cs, xs, name='f(x)')
    epsilon = normal(0, sd=0.1, name='$\\epsilon$', testval=0.1)
    ys = fn(lambda x, y: x + y, exact_ys, epsilon)
    return ys

def test_polynomial_right_way_obs():
    # The right way to write a polynomial with an observation
    cs = normal(2, sd=10, shape=4, name='$c$')
    xs = normal(-2, sd=10, name='$x$', testval=0.1)
    exact_ys = fn(polynomial_fn, cs, xs, name='f(x)')
    epsilon = normal(0, sd=0.1, name='$\\epsilon$', testval=0.1)
    ys_ = fn(lambda x, y: x + y, exact_ys, epsilon)
    ys_obs = normal(ys_, sd=0.01, name="obs_node")
    return ys_obs

def test_AB():
    true_p_A = 0.05
    true_p_B = 0.04
    # unequal sample sizes
    N_A = 1500
    N_B = 750
    observations_A = stats.bernoulli.rvs(true_p_A, size=N_A)
    observations_B = stats.bernoulli.rvs(true_p_B, size=N_B)
    #
    p_A = uniform(0, 1)
    p_B = uniform(0, 1)
    delta = fn(lambda x, y: x - y, p_A, p_B)
    A = flip(p_A)
    B = flip(p_B)
    observe(A, observations_A)
    observe(B, observations_B)
    m = ms_merge([A, B, delta])
    m_draw(m)
    m_pymc3(m)

def test_draw():
    a = uniform(1, 2)
    b = uniform(a, 2, name='$a$')
    c = uniform(b, 2)
    d = uniform(a, b, name='alongname')
    e = uniform(d, c, name='$a$')
    observe(e, np.array([3, 3, 4]))
    observe(e, 10)
    fix(a, T.constant(0.9))
    m_draw(e)
    m_draw_info(e)
    m_pymc3(e)

def test_logistic():
    challenger_data = np.array([(66, 0), (70, 1), (69, 0),
                                (68, 0), (67, 0), (72, 0),
                                (73, 0), (70, 0), (57, 1),
                                (63, 1), (70, 1), (78, 0),
                                (67, 0), (53, 1), (67, 0),
                                (75, 0), (70, 0), (81, 0),
                                (76, 0), (79, 0), (75, 1),
                                (76, 0), (58, 1)])
    temps = challenger_data[:, 0]
    outcomes = challenger_data[:, 1]
    def logistic(x, beta, alpha):
        return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))
    # The model:
    alpha = normal(0, 0.001, name='$\\alpha$')
    beta = normal(0, 0.001, name='$\\beta$')
    t = normal(0, sd=100, name="T")
    p = fn(logistic, t, beta, alpha, name="p")
    outcome = flip(p, name="obs")
    # Fix some variables and observe others
    fix(t, temps)
    observe(outcome, outcomes)
    #
    m_draw_info(outcome)
    trace = model_sample_metropolis(m_pymc3(outcome, True), 120000, sample_args={'burn': 100000, 'thin': 2})
    m_traceplot(trace, outcome, ms=[alpha, beta])
    m_autocorrplot(trace, outcome, ms=[alpha, beta])
    # Evaluating the model at a point (you can specify a subset of the variables if you want)
    print '-17 and 0.26 likelihood:', m_eval(outcome, {alpha: -17, beta: 0.26})

def test_dark_skies():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import numpy as np
    def draw_sky(galaxies):
        """adapted from Vishal Goklani"""
        size_multiplier = 45
        fig = plt.figure(figsize=(10,10))
        #fig.patch.set_facecolor("blue")
        ax = fig.add_subplot(111, aspect='equal')
        n = galaxies.shape[0]
        for i in range(n):
            _g = galaxies[i, :]
            x,y = _g[0], _g[1]
            d = np.sqrt(_g[2]**2 + _g[3]**2)
            a = 1.0/ (1 - d)
            b = 1.0/(1 + d)
            theta = np.degrees(np.arctan2(_g[3], _g[2])*0.5)
            ax.add_patch(Ellipse(xy=(x, y), width=size_multiplier*a, height=size_multiplier*b, angle=theta) )
        ax.autoscale_view(tight=True)
        return fig
    def euclidean_distance(x, y):
        return np.sqrt(((x - y)**2)).sum(axis=1)
    def f_distance(gxy_pos, halo_pos, c):
        # foo_position should be a 2-d numpy array
        # T.maximum() provides our element-wise maximum as in NumPy, but instead for theano tensors
        return T.maximum(euclidean_distance(gxy_pos, halo_pos), c)[:, None]
    def tangential_distance(glxy_position, halo_position):
        # foo_position should be a 2-d numpy array
        delta = glxy_position - halo_position
        t = (2*T.arctan(delta[:, 1]/delta[:, 0]))
        return T.stack([-T.cos(t), -T.sin(t)], axis=1)
    n_sky = 3  # choose a file/sky to examine.
    data = np.genfromtxt("pymc3_joy/Training_Sky%d.csv" % (n_sky),
                         dtype=None,
                         skip_header=1,
                         delimiter=",",
                         usecols=[1, 2, 3, 4])
    print("Data on galaxies in sky %d." % n_sky)
    print("position_x, position_y, e_1, e_2 ")
    print(data[:3])
    plt.title("Galaxy positions and ellipcities of sky %d." % n_sky)
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    # The model:
    mass_large = uniform(40, 180, name="mass_large")
    halo_position = uniform(0, 4200, shape=(1,2), name="halo_position")
    mean = fn(lambda m, p: m / f_distance(T.as_tensor(data[:, :2]), p, 240) * tangential_distance(T.as_tensor(data[:, :2]), p),
              mass_large, halo_position,
              name="mean_location")
    ellpty = normal(mean, 1./0.05, name="ellipcity")
    observe(ellpty, data[:, 2:])
    m_draw(ellpty)
    #
    trace = model_sample_advi_nuts(m_pymc3(ellpty), 5000, advi_args={'n': 50000})
    t = m_trace_variable(trace, ellpty, halo_position).reshape(5000, 2)
    draw_sky(data)
    plt.title("Galaxy positions and ellipcities of sky %d." % n_sky)
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.scatter(t[:, 0], t[:, 1], alpha=0.015, c="r")
    plt.xlim(0, 4200)
    plt.ylim(0, 4200)
