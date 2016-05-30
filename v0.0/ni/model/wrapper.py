import ni.tools.project as project
import ni.config
import create_splines as cs
import create_design_matrix_vk as cdm
import designmatrix
import backend_elasticnet as backend_model
import backend_glm
import backend_elasticnet
import statsmodels.api as sm
import statsmodels.genmod.families.family
from copy import copy
import numpy as np


class Configuration(ni.tools.pickler.Picklable):

    """
    """

    def __init__(self, c=False):
        self.backend_config = False
        self.backend = ni.config.get('model.ip.backend', "glm")
        if self.backend == 'glm':
            self.name = "Generic GLM Model"
        elif self.backend == 'elasticnet':
            self.name = "Generic Elastic Net Model"
        else:
            self.name = "Generic Model"
        if type(c) == str:
            self.loads(c)
        if type(c) == dict:
            self.__dict__.update(c)
            self.eval_dict()
        if isinstance(c, Configuration):
            self.loads(c.dumps())

    def __str__(self):
        return "ni.model.wrapper.Configuration(\""+ni.tools.pickler.dumps(self.__dict__).replace('"', '\\"')+"\")"


class FittedModel(ni.tools.pickler.Picklable):

    """
    When initialized via Model.fit() it contains a copy of the configuration, a link to the model it was fitted from and fitting parameters:

            FittedModel. **fit**

                    modelFit Output

            FittedModel. **design**

                    The DesignMatrix used. Use *design.matrix* for the actual matrix or design.get('...') to extract only the rows that correspond to a keyword.


    """

    def __init__(self, model):

        if type(model) == str or type(model) == dict:
            self.loads(model)
            if type(self.beta) == str:
                self.beta = eval(self.beta.replace("np.ndarray", "np.array"))
        else:
            if isinstance(model, Model):
                self.model = model
            else:
                raise Exception("Initialized without a model.")

    @property
    def complexity(self):
        """ returns the length of the parameter vector """
        try:
            return len(self.beta)
        except:
            return False

    def getParams(self):
        """ returns the parameters of each design matrix component as a list """
        return [np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)], 1) for h in np.unique(self.design.header)]

    def getPvalues(self):
        """ returns pvalues of each component as a dictionary """
        return dict((h, self.statistics.pvalues[self.design.getIndex(h)]) for h in np.unique(self.design.header))

    def pvalues_by_component(self):
        """ returns pvalues of each component as a dictionary """
        return dict((h, self.statistics.pvalues[self.design.getIndex(h)]) for h in np.unique(self.design.header))

    def plotParams(self, x=-1):
        """ plots the parameters and returns a dictionary of figures """
        figs = {}
        for h in np.unique(self.design.header):
            fig = pl.figure()
            pl.plot(
                np.sum(self.design.get(h)[:x] * self.beta[self.design.getIndex(h)], 1))
            pl.title(h)
            figs[h] = fig
        return figs

    def plot_prototypes(self):
        """ plots each of the components as a prototype (sum of fitted b-splines) and returns a dictionary of figures """
        figs = {}
        for h in np.unique(self.design.header):
            fig = pl.figure()
            splines = self.design.get(h)
            if 'rate' in h:
                pl.plot(np.sum(self.design.get(
                    h)[:self.design.trial_length] * self.beta[self.design.getIndex(h)], 1))
                pl.title(h)
            elif len(splines.shape) == 1 or (splines.shape[0] == 1):
                pl.plot(
                    np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)], 1), 'o')
                pl.title(h)
            elif len(splines.shape) == 2:
                pl.plot(
                    np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)], 1))
                pl.title(h)
            elif len(splines.shape) == 3:
                slices = np.zeros(splines.shape)
                for (i, ind) in zip(range(splines.shape[0]), self.design.getIndex(h)):
                    slices[i, :, :] = splines[i, :, :] * self.beta[ind]
                pl.imshow(slices.sum(axis=0), cmap='jet')
                figs[h + '_sum'] = fig
                fig = pl.figure()
                for i in range(len(slices)):
                    pl.subplot(
                        np.ceil(np.sqrt(slices.shape[0])), np.ceil(np.sqrt(slices.shape[0])), i+1)
                    pl.imshow(slices[i], vmin=np.percentile(
                        slices, 1), vmax=np.percentile(slices, 99), cmap='jet')
                pl.suptitle(h)
                figs[h] = fig
            else:
                pl.plot(
                    np.sum(self.design.get(h) * self.beta[self.design.getIndex(h)], 1))
                pl.title(h)
            figs[h] = fig
        return figs

    def prototypes(self):
        """ returns a dictionary with a prototype (numpy.ndarray) per component """
        prot = {}
        for h in np.unique(self.design.header):
            splines = self.design.get(h)
            if len(splines.shape) == 1:
                prot[h] = self.design.get(
                    h) * self.beta[self.design.getIndex(h)]
            elif len(splines.shape) == 2:
                prot[h] = np.sum(
                    self.design.get(h) * self.beta[self.design.getIndex(h)], 1)
            elif len(splines.shape) == 3:
                prot[h] = np.zeros(splines.shape)
                for (i, ind) in zip(range(splines.shape[0]), self.design.getIndex(h)):
                    prot[h][i, :, :] = splines[i, :, :] * self.beta[ind]
        return prot

    def firing_rate_model(self):
        """ returns a time series which contains the rate and constant component """
        rate_design = self.design.getIndex('rate')
        return self.design.get('rate')[:self.design.trial_length, :]*self.beta[rate_design]

    def plot_firing_rate_model(self):
        """ returns a time series which contains the rate and constant component """
        return plot(self.firing_rate_model())

    def generate(self, bins=-1):
        """
        Generates new spike trains from the extracted staistics

        This function only uses rate model and autohistory. For crosshistory dependence, use :mod:`ip_generator`.

                **bins**

                        How many bins should be generated (should be multiples of trial_length)

        """
        spikes = []
        if bins < 0:
            bins = self.design.trial_length
        prototypes = self.getPrototypes()
        ps = []
        # *self.firing_rate*self.firing_rate/abs(np.mean(prototypes['autohistory']))
        autohistory = prototypes['autohistory']
        rate = self.firing_rate_model().sum(1) + prototypes['constant']
        time = np.zeros(bins)
        for i in range(self.design.trial_length):
            rand = np.random.rand()
            p = rate[i]
            ps.append(p)
            if rand < self.family_fitted_function(p):
                time[i] = 1
                spikes.append(i)
                kernel_end = np.min([autohistory.shape[0], len(rate) - i])
                rate[i:i+kernel_end] = rate[i:i+kernel_end] + \
                    autohistory[:kernel_end]
        return (spikes, time, np.array(ps))

    def family_fitted_function(self, p):
        """
                only implemented family: Binomial
        """
        return sm.families.Binomial().fitted(p)

    def predict(self, data):
        """
        Using the model this will predict a firing probability function according to a design matrix.
        """
        if isinstance(data, Data) or isinstance(data, ni.data.data.Data):
            dm = self.design.combine(data)
        else:
            dm = data

        return self.model.backend.predict(self.beta, dm)

    def compare(self, data):
        """
        Using the model this will predict a firing probability function according to a design matrix.

        Returns:

                **Deviance_all**: dv, 
                **LogLikelihood_all**: ll, 
                **Deviance**: dv/nr_trials, 
                **LogLikelihood**: ll/nr_trials, 
                **llf**: Likelihood function over time 
                **ll**: np.sum(ll)/nr_trials
        """
        return self.model.compare(data, self.predict(data))

    def dumps(self):
        """ see :mod:`ni.tools.pickler` """
        return ni.tools.pickler.dumps({'beta': self.beta, 'model': self.model, 'statistics': self.statistics})

    def html_view(self):
        """ see :mod:`ni.tools.html_view` """
        view = View()
        model_prefix = self.model.name + "/"
        view.add(model_prefix + "#2/beta", self.beta)
        for key in self.configuration.__dict__:
            view.add(model_prefix + "#3/tabs/Configuration/table/" +
                     key, self.configuration.__dict__[key])
        view.add(model_prefix + "#3/tabs/Design", "")
        prot_nr = 0
        prototypes = self.plot_prototypes()
        for p in prototypes:
            prot_nr = prot_nr + 1
            view.savefig(
                model_prefix + "#3/tabs/Prototypes/tabs/"+str(p), fig=prototypes[p])
        prototypes = self.prototypes()
        if 'autohistory2d' in prototypes and 'autohistory' in prototypes:
            with view.figure(model_prefix + "#3/tabs/Prototypes/tabs/autohistory_2d+autohistory"):
                for i in range(prototypes["autohistory2d"].shape[2]):
                    pl.plot(sum(prototypes["autohistory2d"], 0)[
                            i, :]+prototypes["autohistory"], 'g:')
                pl.plot(prototypes["autohistory"], 'b-')
        for c in self.design.components:
            if type(c) != str:
                with view.figure(model_prefix + "#3/tabs/Design/#2/tabs/"+c.header+"/#2/Splines"):
                    splines = c.getSplines()
                    if len(splines.shape) == 1:
                        pl.plot(splines, '-o')
                    elif len(splines.shape) == 2:
                        if splines.shape[0] == 1 or splines.shape[1] == 1:
                            pl.plot(splines, '-o')
                        else:
                            pl.plot(splines)
                    elif len(splines.shape) == 3:
                        for i in range(splines.shape[0]):
                            pl.subplot(
                                np.ceil(np.sqrt(splines.shape[0])), np.ceil(np.sqrt(splines.shape[0])), i+1)
                            pl.imshow(
                                splines[i, :, :], interpolation='nearest')
            else:
                view.add(
                    model_prefix + "#3/tabs/Design/#2/tabs/##/component", str(c))
        return view


class Model(ni.tools.pickler.Picklable):

    def __init__(self, configuration=None):
        """
        A wrapper for glm or elastic net models providing a unified interface to fit an equation:

                $x = g( beta \dot dm )$

        (conventionally also $y = beta \dot X$, but here x is the *dependent* rather than the independent variable)

        Uses one of two backends :py:module:`ni.model.backend_glm` and :py:module:`ni.model.backend_elasticnet`
        Select backend by setting `model.backend = 'glm'` or `model.backend = 'elasticnet'`.

        """
        if configuration == None:
            configuration = {}
        if type(configuration) == dict:
            configuration = Configuration(configuration)
        self.configuration = configuration
        self.name = self.configuration.name
        self.loads(configuration)
        if self.configuration.backend == "glm":
            self._backend = backend_glm
        elif self.configuration.backend == "elasticnet":
            self._backend = backend_elasticnet
        else:
            raise Exception(
                "Backend '"+self.configuration.backend+"' is not a Model backend.")
        #self.rate_splines = cs.create_splines_linspace(nr_bins, self.configuration.knots_rate, 0)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, b):
        if b == "glm":
            self._backend = backend_glm
        elif b == "elasticnet":
            self._backend = backend_elasticnet

    def predict(self, beta, data):
        """
        will predict a firing probability function according to a design matrix.
        """
        return self.backend.predict(beta, dm)

    def compare(self, data, p, nr_trials=1):
        """
        will compare a timeseries of probabilities `p` to a binary timeseries or Data instance `data`.

        Returns:

                **Deviance_all**: dv, 
                **LogLikelihood_all**: ll, 
                **Deviance**: dv/nr_trials, 
                **LogLikelihood**: ll/nr_trials, 
                **llf**: Likelihood function over time 
                **ll**: np.sum(ll)/nr_trials
        """
        binomial = statsmodels.genmod.families.family.Binomial()
        #x = self.x(data)
        x = data.squeeze()
        p = p.squeeze()
        p[p <= 0] = 0.000000001
        dv = binomial.deviance(x, p)
        ll_bin = x * np.log(p) + (1-x)*np.log(1-p)
        ll_bin[np.isnan(ll_bin)] = np.min(ll_bin)
        ll = binomial.loglike(x, p)
        if isinstance(data, ni.data.data.Data):
            nr_trials = data.nr_trials
        return {'Deviance': dv/nr_trials, 'Deviance_all': dv, 'LogLikelihood': ll/nr_trials, 'LogLikelihood_all': ll, 'llf': ll_bin, 'll': np.sum(ll_bin)/nr_trials}

    def fit(self, x=None, dm=None, beta=None, nr_trials=None):
        """
        Fits the model

                **x** `np.array` vector of spike data to be predicted
                **dm** `np.array` matrix of size |y| x |\\beta| (or the other way around?)


        example::



        """
        fittedmodel = FittedModel(self)
        fittedmodel.configuration = copy(self.configuration)

        firing_rate = np.mean(x)/1000.0
        fittedmodel.firing_rate = firing_rate
        fittedmodel.trial_length = dm.shape[1]

        if type(dm) == dict and 'dm' in dm.keys():
            dm = dm['dm']

        w = np.where(dm.transpose())[0]
        cnt = [np.sum(w == i) for i in range(dm.shape[1])]

        if sum(np.array(cnt) >= dm.shape[0]) > 0:
            print "!! "+str(sum(np.array(cnt) == dm.shape[0])) + " Components are only 0. \n"+str(sum(np.array(cnt) <= dm.shape[0]*0.1))+" are mostly 0. "+str(sum(np.array(cnt) <= dm.shape[0]*0.5))+" are half 0."
        else:
            print ""+str(sum(np.array(cnt) == dm.shape[0])) + " Components are only 0. \n"+str(sum(np.array(cnt) <= dm.shape[0]*0.1))+" are mostly 0. "+str(sum(np.array(cnt) <= dm.shape[0]*0.5))+" are half 0."
        zeroed_components = [i for i in range(len(cnt)) if cnt[i] == 0]

        backend_config = self.configuration.backend_config
        fittedmodel.backend_model = self.backend.Model(backend_config)
        if beta is None:
            # do an actual fit
            fit = fittedmodel.backend_model.fit(x.squeeze(), dm)
            fittedmodel.beta = fit.params
        else:
            # only load the precomputed result
            fit = self.backend.Fit(f=None, m=fittedmodel.backend_model)
            fit.params = beta
            fittedmodel.params = beta
            fittedmodel.beta = beta
        fittedmodel.fit = fit
        if "llf" in fit.statistics:
            fit.statistics["llf_all"] = fit.statistics["llf"]
            if hasattr(data, 'nr_trials'):
                fit.statistics["llf"] = fit.statistics["llf"]/data.nr_trials
            elif nr_trials is not None:
                fit.statistics["llf"] = fit.statistics["llf"]/nr_trials
        fittedmodel.statistics = fit.statistics
        return fittedmodel
    # def fit_with_design_matrix(self, fittedmodel, spike_train_all_trial, dm):
    #        """ internal function """
    #        fittedmodel.fit = fittedmodel.backend_model.fit(spike_train_all_trial, dm)
    #        return fittedmodel

    def html_view(self):
        """ see :mod:`ni.tools.html_view` """
        view = View()
        model_prefix = self.name + "/"
        for key in self.configuration.__dict__:
            view.add(model_prefix + "#3/tabs/Configuration/table/" +
                     key, self.configuration.__dict__[key])
        return view
