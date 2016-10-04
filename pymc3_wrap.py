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
# If you want to hack in the notebook sprinkle reload(pymc3_extra) and
# reload(pymc3_wrap) in your code, edit the file, and rerun the cell.

# Wrap up as a monad. m_merge is >>. const is return. fix is also a
# form of return since it replaces a node with a constant.
#
# model_ functions operate on pymc3 models.
# m_ functions operate on wrapped models, ie. those of class M.

# FIXME What's going on with observations? How do we observe two nodes
# jointly?
# FIXME Automatically lift to theano? Or at least do it manually.
# FIXME Check shapes to avoid strange errors
# FIXME Add observations
# FIXME Don't subscript unique names
# FIXME marginalize values
# FIXME many plots
# FIXME many sampling mechanisms
# FIXME show shape in the nodes
# FIXME Get rid of v.observed and v.fixed and compute off of the distributions
# FIXME Time series models
# FIXME The default testvals are shitty for hierarchical models. Randomize them.
# FIXME minibatch ADVI 
# FIXME pm.df_summary(trace)
# FIXME Find inputs and allow a call
# FIXME To wrap up:
# https://github.com/pymc-devs/pymc3/blob/master/docs/source/notebooks/cox_model.ipynb
#  _ = pm.traceplot(traces_signoise[-1000:], figsize=(12,len(traces_signoise.varnames)*1.5),
#            lines={k: v['mean'] for k, v in pm.df_summary(traces_signoise[-1000:]).iterrows()})
# https://pymc-devs.github.io/pymc3/notebooks/posterior_predictive.html

# Runtime debugging:
# FIXME Draw 1 sample and 1 good sample
# FIXME Set some inputs and draw 1 sample
# FIXME Show all intermediate values
# FIXME Treat the model as a function and show intermediate values
# FIXME Call function interface

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
            s.append(' {} {}[{}] {} observed={} fixed={}'.format(name, v.var_type, v.shape, v.pymc3fn, v.observed, v.fixed))
        def fedge(location, vin, namein, valin, v, name, val):
            return ()
        def fargs(v, name, args, kwargs):
            return ()
        m_walk(fone, fedge, fargs, self).keys()
        return '\n'.join(s)


class V():
    def __init__(self, pymc3fn, args, kwargs, shape, var_type, var_kind, maybe_name, observed, fixed, printing):
        # var_type is continuous, discrete, potential, etc.
        # var_kind is unif, N, Dir, Beta, etc.
        self.pymc3fn = pymc3fn
        self.args = args
        self.kwargs = kwargs
        self.shape = shape
        self.var_type = var_type
        self.var_kind = var_kind
        self.maybe_name = maybe_name
        self.observed = observed
        self.fixed = fixed
        self.printing = printing
        if self.observed and self.fixed:
            raise ValueError("Can't be both fixed and observed")

# Misc functions (should likely go elsewhere)


def deduplicate(lst):  # keeps order
    uniq = []
    for i in lst:
        if i not in uniq:
            uniq.append(i)
    return uniq


def replace_dollar_signs(s, n):
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
def m_pymc3(m):
    with pm.Model() as model:
        def fone(v, name):
            return v
        def fedge(location, vin, namein, valin, v, name, val):
            return ()
        def fargs(v, name, args, kwargs):
            r = v.pymc3fn(name, *args, **kwargs)
            if v.printing:
                r = T.printing.Print(name)(r)
            return r
        m_walk(fone, fedge, fargs, m)
    return model


# Distributions


def m_wrap(f, var_type, var_kind, *args, **kwargs):
    #    https://github.com/pymc-devs/pymc3/issues/829
    name = kwargs.pop('name', False)    
    v = V(f,
          args,
          kwargs,
          kwargs.get('shape', 1),
          var_type,
          var_kind,
          name,
          'observed' in kwargs,
          False,
          False)
    ms = dependencies_ms(args, kwargs)
    ms.append(M([v], v))
    return ms_merge(ms)


def uniform(*args, **kwargs):
    #([lower, upper, transform])	Continuous uniform log-likelihood.
    return m_wrap(pm.Uniform, 'continuous', '$\mathcal{U}$', *args, **kwargs)


def flat(*args, **kwargs):
    #(*args, **kwargs)	Uninformative log-likelihood that returns 0 regardless of the passed value.
    return m_wrap(pm.Flat, 'continuous', 'flat', *args, **kwargs)


def normal(*args, **kwargs):
    #(*args, **kwargs)	Univariate normal log-likelihood.
    return m_wrap(pm.Normal, 'continuous', '$\mathcal{N}$', *args, **kwargs)


def beta(*args, **kwargs):
    #([alpha, beta, mu, sd])	Beta log-likelihood.
    return m_wrap(pm.Beta, 'continuous', 'Beta', *args, **kwargs)


def exponential(*args, **kwargs):
    #(lam, *args, **kwargs)	Exponential log-likelihood.
    return m_wrap(pm.Exponential, 'continuous', 'Exp', *args, **kwargs)


def laplace(*args, **kwargs):
    #(mu, b, *args, **kwargs)	Laplace log-likelihood.
    return m_wrap(pm.Laplace, 'continuous', 'Lap', *args, **kwargs)


def studentT(*args, **kwargs):
    #(nu[, mu, lam, sd])	Non-central Student’s T log-likelihood.
    return m_wrap(pm.StudentT, 'continuous', 't-dist', *args, **kwargs)


def cauchy(*args, **kwargs):
    #(alpha, beta, *args, **kwargs)	Cauchy log-likelihood.
    return m_wrap(pm.Cauchy, 'continuous', 'cauchy', *args, **kwargs)


def halfCauchy(*args, **kwargs):
    #(beta, *args, **kwargs)	Half-Cauchy log-likelihood.
    return m_wrap(pm.HalfCauchy, 'continuous', '1/2-cauchy', *args, **kwargs)


def gamma(*args, **kwargs):
    #([alpha, beta, mu, sd])	Gamma log-likelihood.
    return m_wrap(pm.Gamma, 'continuous', 'Gamma', *args, **kwargs)


def weibull(*args, **kwargs):
    #(alpha, beta, *args, **kwargs)	Weibull log-likelihood.
    return m_wrap(pm.Weibull, 'continuous', 'Weibull', *args, **kwargs)


def studentTpos(*args, **kwargs):
    #(*args, **kwargs)	
    return m_wrap(pm.StudentTpos, 'continuous', 't-dist+', *args, **kwargs)


def logNormal(*args, **kwargs):
    #([mu, sd, tau])	Log-normal log-likelihood.
    return m_wrap(pm.Lognormal, 'continuous', 'log$\mathcal{N}$', *args, **kwargs)


def chiSquared(*args, **kwargs):
    #(nu, *args, **kwargs)	χ2χ2 log-likelihood.
    return m_wrap(pm.ChiSquared, 'continuous', '$\chi^{2}$', *args, **kwargs)


def halfNormal(*args, **kwargs):
    #([sd, tau])	Half-normal log-likelihood.
    return m_wrap(pm.HalfNormal, '$\mathcal{N}^{+}$', *args, **kwargs)


def wald(*args, **kwargs):
    #([mu, lam, phi, alpha])	Wald log-likelihood.
    return m_wrap(pm.Wald, 'continuous', 'wald', *args, **kwargs)

    
def pareto(*args, **kwargs):
    #(alpha, m, *args, **kwargs)	Pareto log-likelihood.
    return m_wrap(pm.Pareto, 'continuous', 'pareto', *args, **kwargs)


def inverseGamma(*args, **kwargs):
    #(alpha[, beta])	Inverse gamma log-likelihood, the reciprocal of the gamma distribution.
    return m_wrap(pm.InverseGamma, 'continuous', '$\text{Gamma}^-1$', *args, **kwargs)


def exGaussian(*args, **kwargs):
    #(mu, sigma, nu, *args, **kwargs)	Exponentially modified Gaussian log-likelihood.
    return m_wrap(pm.ExGaussian, 'continuous', 'Exp(Gamma)', *args, **kwargs)


def binomial(*args, **kwargs):
    #(n, p, *args, **kwargs)	Binomial log-likelihood.
    return m_wrap(pm.Binomial, 'discrete', 'Binomial', *args, **kwargs)


def betaBinomial(*args, **kwargs):
    #(alpha, beta, n, *args, **kwargs)	Beta-binomial log-likelihood.
    return m_wrap(pm.BetaBinomial, 'discrete', 'BetaBin', *args, **kwargs)


def bernoulli(*args, **kwargs):
    #(p, *args, **kwargs)	Bernoulli log-likelihood
    return m_wrap(pm.Bernoulli, 'discrete', 'Bern', *args, **kwargs)


def poisson(*args, **kwargs):
    #(mu, *args, **kwargs)	Poisson log-likelihood.
    return m_wrap(pm.Poisson, 'discrete', 'Poisson', *args, **kwargs)


def negativeBinomial(*args, **kwargs):
    #(mu, alpha, *args, **kwargs)	Negative binomial log-likelihood.
    return m_wrap(pm.NegativeBinomial, 'discrete', 'NegBin', *args, **kwargs)


def constantDist(*args, **kwargs):
    #(c, *args, **kwargs)	Constant log-likelihood.
    return m_wrap(pm.ConstantDist, 'discrete', 'Const', *args, **kwargs)


def zeroInflatedPoisson(*args, **kwargs):
    #(theta, psi, *args, **kwargs)	Zero-inflated Poisson log-likelihood.
    return m_wrap(pm.ZeroInflatedPoisson, 'discrete', 'ZIPos', *args, **kwargs)


def discreteUniform(*args, **kwargs):
    #(lower, upper, *args, **kwargs)	Discrete uniform distribution.
    return m_wrap(pm.DiscreteUniform, 'discrete', 'disc', *args, **kwargs)


def geometric(*args, **kwargs):
    #(p, *args, **kwargs)	Geometric log-likelihood.
    return m_wrap(pm.Geometric, 'discrete', 'Geom', *args, **kwargs)


def categorical(*args, **kwargs):
    #(p, *args, **kwargs)	Categorical log-likelihood.
    return m_wrap(pm.Categorical, 'discrete', 'cat', *args, **kwargs)


def mvNormal(*args, **kwargs):
    #(mu[, cov, tau])	Multivariate normal log-likelihood.
    return m_wrap(pm.MvNormal, 'continuous', 'mv$\mathcal{N}$', *args, **kwargs)


def wishart(*args, **kwargs):
    #(n, V, *args, **kwargs)	Wishart log-likelihood.
    return m_wrap(pm.Wishart, 'continuous', 'Wishart', *args, **kwargs)


def lkjCorr(*args, **kwargs):
    #(n, p, *args, **kwargs)	The LKJ (Lewandowski, Kurowicka and Joe) log-likelihood.
    return m_wrap(pm.LKJCorr, 'continuous', 'LKJ', *args, **kwargs)


def multinomial(*args, **kwargs):
    #(n, p, *args, **kwargs)	Multinomial log-likelihood.
    return m_wrap(pm.Multinomial, 'discrete', 'multi', *args, **kwargs)


def dirichlet(*args, **kwargs):
    #(a[, transform])	Dirichlet log-likelihood.
    return m_wrap(pm.Dirichlet, 'discrete', 'Dir', *args, **kwargs)


def flip(*args, **kwargs):
    return bernoulli(*args, **kwargs)


# Special nodes:

# We don't expose pm.Deterministic only fn.
# Instead of pm.Deterministic("A", a-b)
# You want fn(-, a, b)
def fn(fun, *args, **kwargs):
    def f(name, *args, **kwargs):
        return pm.Deterministic(name, fun(*args), **kwargs)
    return m_wrap(f, 'deterministic', 'fn', *args, **kwargs)


def const(val, **kwargs):
    def f(name, *args, **kwargs):
        return pm.Deterministic(name, lambda x: val, **kwargs)
    return m_wrap(f, 'deterministic', 'fn', (), **kwargs)


def density(fun, *args, **kwargs):
    def f(name, *args, **kwargs):
        return pm.DensityDist(name, fun, *args, **kwargs)
    return m_wrap(f, 'density', 'density', *args, **kwargs)


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
                  '$psi$',
                  name,
                  'observed' in kwargs,
                  False,
                  False))
    return m


# Manipulate nodes after the fact


def printing(m, toggle=True):
    m.output.printing = True
    return m


def observe(m, data):
    # We rely on the fact that Vs are shared between models and _never_ rebuilt or copied
    m.output.kwargs['observed'] = data
    m.output.observed = True
    return m


def fix(m, data):
    # We rely on the fact that Vs are shared between models and _never_ rebuilt or copied
    #
    # pymc3 doesn't have the notion of a fixed variable. So instead we
    # swap out the distribution for a constant deterministic variable.
    v = m.output
    v.pymc3fn = pm.Deterministic
    v.args = (data,)
    v.kwargs = {}
    v.var_type = 'deterministic'
    if v.observed:
        raise ValueError("You can't fix an observed variable. This doesn't make sense")
    v.fixed = 1
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
        g.node(name, label=replace_dollar_signs(label, 2))
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
    pgm = daft.PGM(shape=[max(xs(layout))+1, max(ys(layout))+1],
                   origin=[-0.1,-0.1],
                   grid_unit=2,
                   node_unit=1)
    def fone(v, name):
        label = name
        if flabel:
            label = flabel(v, name)
        pgm.add_node(daft.Node(name, label, layout[v][0], layout[v][1], observed=v.observed, fixed=v.fixed, aspect=layout[v][2]))
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


# Runtime debugging


def m_profile(m):
    return model_profile(m_pymc3(m))


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
    # variable for some reason.
    #
    # FIXME why can't we observe deterministic variables?
    ys = normal(exact_ys, sd=0.1, name='$\epsilon$', testval=0.1)
    return ys


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
    delta = fn(lambda x, y: x-y, p_A, p_B)
    A = flip(p_A)
    B = flip(p_B)
    observe(A, observations_A)
    observe(B, observations_B)
    m = ms_merge([A, B, delta])
    m_draw(m)
    m_pymc3(m)


def test_draw():
    a = uniform(1,2)
    b = uniform(a,2,name='$a$')
    c = uniform(b,2)
    d = uniform(a,b,name='alongname')
    e = uniform(d,c,name='$a$')
    observe(e,np.array([3,3,4]))
    observe(e,10)
    fix(a,T.constant(0.9))
    m_draw(e)
    m_draw_info(e)
    m_pymc3(e)
