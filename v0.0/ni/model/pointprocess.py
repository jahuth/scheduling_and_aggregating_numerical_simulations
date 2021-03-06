"""
.. module:: ni.model.pointprocess
   :platform: Unix
   :synopsis: Storing Point Process Data

.. moduleauthor:: Jacob Huth

"""
import ni
import numpy as np
import pylab
import scipy.ndimage
import pandas as pd
import warnings

class PointProcess:
	"""
	A Point Process container.

	Usually generated by loading from a file or via :py:func:`ni.model.pointprocess.createPoisson`		
	"""
	def __init__(self,dimensionality):
		self.initialized = False
		self.range = (0,0)
		self.dimensionality = dimensionality
		self.spikes = []
		self.frate = np.array(0)
	def addSpike(self,t):
		"""
		adds a spike to the point process, if it falls in the allowed range.
		"""
		if type(t) == int:
			if t > self.range[0] and t < self.range[1]:
				self.spikes.append(t)
		else:
			if (t > self.range[0]).all and (t < self.range[1]).all:
				self.spikes.append(t)
	def getProbability(self,t_from,t_to):
		"""
		Undocumented
		"""
		return 0	
	def getCounts(self,time_scale=1.0):
		"""
		Gives a (in most cases binary) time series of the point process.
		"""
		ba = np.zeros(int(self.range[1]*time_scale))
		for s in self.spikes:
			ba[int(s*time_scale)] = ba[int(s*time_scale)] + 1
		return ba.astype(float) 
	def plot(self,y=0,marker='|'):
		"""
		Plots the pointprocess as points at line `y`.

		`marker` determines the color and shape of the marker. Default is a vertical line '|'
		"""
		pylab.plot(self.spikes,[y]*len(self.spikes),marker)
	def plotGaussed(self,width):
		"""
		Plots the pointprocess as a smoothed time series
		"""
		pylab.plot(scipy.ndimage.gaussian_filter(self.getCounts(),width))

def getCounts(spikes):
	"""
	Gives back an array of spike counts from an array of spike times. If the output is suppsed to be a Binomial, use `getBinary` instead.
	"""
	ba = np.array([0]*(np.max(spikes)+1))
	for s in spikes:
		ba[s] = ba[s] + 1
	return ba.astype(float)

def getBinary(spikes, min_length = 1):
	"""
	Gives back a binary array from an array of spike times. The maximum for each bin is 1.
	"""
	if len(spikes) < 1:
		return np.array([0]*min_length)
	ba = np.array([0]*(np.max([np.max(spikes)+1,min_length])))
	collisions = 0
	for s in spikes:
		if ba[s] > 0:
			collisions = collisions + 1
		ba[s] = 1
	if collisions > 0:
		warnings.warn("Warning: " +str(collisions)+ " Spike collision(s) occured. You might want to increase bin spacing.",UserWarning)
	return ba.astype(float)

def reverse_correlation(spikes_a,spikes_b=False):
	isi = interspike_interval(spikes_a,spikes_b)
	rev = np.zeros(np.max(isi)+1)
	for i in isi:
		rev[i] = rev[i] + 1
	rev = rev / len(isi)
	return rev


def interspike_interval(spikes_a,spikes_b=False):
	if type(spikes_b) == bool:
		spikes_b = spikes_a
	isi = []
	for s in np.where(spikes_a)[0]:
		intervals = np.where(spikes_b[(s+1):])[0]
		if len(intervals) > 0:
			isi.append(intervals[0])
	return isi


def PointProcessFromSpikeTimes(times):
	#if np.size(times,1) > 1:
	#	return [PointProcessFromSpikeTimes(times[i]) for i in xrange(np.size(times,0))]
	pp = PointProcess(1)
	pp.range = (0,np.max(times)+1)
	pp.spikes = times
	return pp

def createPoisson(p,l):
	"""
	This generates a spike sequence of length `l` according to either a fixed firing rate `p`, or a repeated sequence of firing rates if `type(p) == np.ndarray`.

	It creates a :py:class:`ni.model.pointprocess.PointProcess`

	Example 1:

	.. testcode::

		p1 = ni.model.pointprocess.createPoisson(0.1,1000)
		p1.plotGaussed(20)
		plot(p1.frate)

	.. image:: _static/p1_out_smooth.png

	.. testcode::

		p2 = ni.model.pointprocess.createPoisson(sin(numpy.array(range(0,200))*0.01)*0.5- 0.2,1000)
		p2.plot()
		p2.plotGaussed(10)
		
	.. image:: _static/p2_out.png

	.. testcode::

		p2.plotGaussed(20)
		plot(p2.frate)
		
	.. image:: _static/p2_out_smooth.png

	Example with multiple channels:

	.. testcode::

		frate = (numpy.array(range(0,200))*0.001)*0.2+0.01
		channels = 9

		dists = [ni.model.pointprocess.createPoisson(frate,1000) for i in range(0,channels)]
		#for i in range(0,9): dists[i].plotGaussed(10)
		import itertools
		spks = np.array([dists[i].getCounts() for i in range(0,channels) for j in range(0,99) ])
		imshow(-1*spks)
		set_cmap('gray')

	Will generate:

	.. testoutput::

		(A plot of spikes)
	.. image:: _static/multichannel_spike_plot.png


	.. testcode::

		ni.model.pointprocess.plotGaussed(np.array([dists[i].getCounts() for i in range(0,channels)]).mean(axis=0),20)
		plot(dists[0].frate)

	.. image:: _static/multichannel_sum_gaussed.png

	"""
	pp = PointProcess(1)
	pp.range = (0,l)
	if type(p) == np.ndarray:
		pp.frate = np.zeros(l)
		for i in range(0,l):
			pp.frate[i] = p[i % len(p)]
	else:
		pp.frate = np.ones(l) * p
	for i in range(0,l):
		if np.random.rand() < pp.frate[i % len(pp.frate)]:
			pp.addSpike(i)
	return pp

def plotGaussed(data,width):
	"""
	.. testcode::

		p2 = ni.model.pointprocess.createPoisson(sin(numpy.array(range(0,200))*0.01)*0.5- 0.2,1000)
		p2.plot()
		p2.plotGaussed(10)
		
	.. image:: _static/p2_out.png
	"""
	pylab.plot(scipy.ndimage.gaussian_filter(data,width))

def plotMultiSpikes(spikes):
	"""
	* `spikes` is a binary 2d matrix 

	Generates something like:

	.. image:: _static/plotMultiSpikes.png

	"""
	#[plotGaussed(spikes[i]+i,1)  for i in range(0,len(spikes.T))]
	[pylab.plot(np.where(spikes[i]>0)[0], spikes[i][spikes[i]>0]+i+0.5,'|', markersize=12)  for i in range(0,len(spikes))]

class SimpleFiringRateModel:
	"""
		Uses just the firing rate as a predictor
	"""
	def __init__(self):
		self.frate = 0
	def fit(self,data):
		if isinstance(data, ni.Data):
			self.frate = data.firing_rate()
		else:
			self.frate = np.mean(data)
		return self
	def predict(self,Data):
		self.fit(Data)
		return [self.frate] * len(Data)
	def loglikelihood(self,Data,Prediction): 
		#return [np.sum(Data) * np.log(self.frate) - self.frate * len(Data)]*len(Data)
		return [np.sum(Data) * np.log(self.frate) + np.log(1-self.frate) * len(Data)]*len(Data)
	def compare(self,Data,Prediction): 
		#return [np.sum(Data) * np.log(self.frate) - self.frate * len(Data)]*len(Data)
		return {'LogLikelihood': [np.sum(Data) * np.log(self.frate) + np.log(1-self.frate) * len(Data)]*len(Data)}
