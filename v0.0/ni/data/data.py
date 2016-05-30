"""
.. module:: ni.data.data
   :platform: Unix
   :synopsis: Storing Point Process Data

.. moduleauthor:: Jacob Huth

.. todo::
		Use different internal representations, depending on use. Ie. Spike times vs. binary array

.. todo::
		Lazy loading and prevention from data duplicates where unnecessary. See also: `indexing view versus copy <http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy>`_


Using the ni.Data data structures
------------------------------------

The `Data` class is supposed to be easily accessible to the `ni.` models. They contain an index that separates the time series into different **cells**, **trials** and **conditions**.

**Conditions** are mostly for the users, as they are ignored by the model classes. They should be used to separate data before fitting a model on them, such that only data from a certain subset of trials (ie. one or more experimental conditions) are used for the fit.
If multiple conditions are contained in a dataset that is passed to a model, the model should treat them as additional trials.

**Trials** assume a common time frame ie. that bin 0 of each trial corresponds to the same time relative to a stimulus, such that rate fluctuations can be averaged over trials.

**Cells** signify spike trains that are recorded from different sources (or spike sorted), such that there can be correlations between cells in a certain trail.

The index is hierarchical, as in for each condition there are several trials, which each have several cells.
But since modelling is mainly used to distinguish varying behaviour of the same ensemble of cells, the number of cells in a trial and the number of trials pro condition has to be equal.





Storing Spike Data in Python with Pandas
--------------------------------------------------------

The `pandas package <http://pandas.pydata.org/>`_ allows for easy storage of large data objects in python. The structure that is used by this toolbox is the pandas :py:class:`pandas.MultiIndexedFrame` which is a :py:class:`pandas.DataFrame` / `pandas.DataFrame <http://pandas.pydata.org/pandas-docs/dev/dsintro.html#dataframe>`_ with an Index that has multiple levels.

The index contains at least the levels ``'Cell'``, ``'Trial'`` and ``'Condition'``. Additional Indizex can be used (eg. ``'Bootstrap Sample'`` for Bootstrap Samples), but keep in mind that when fitting a model only ``'Cell'`` and ``'Trial'`` should remain, all other dimensions will be collapsed as more sets of Trials which may be indistinguishable after the fit.


===========  =====  ======= ===================================
Condition    Cell   Trial   *t* (Timeseries of specific trial)
===========  =====  ======= ===================================
0            0      0       0,0,0,0,1,0,0,0,0,1,0...      
0            0      1       0,0,0,1,0,0,0,0,1,0,0...
0            0      2       0,0,1,0,1,0,0,1,0,1,0...
0            1      0       0,0,0,1,0,0,0,0,0,0,0...
0            1      1       0,0,0,0,0,1,0,0,0,1,0...
...          ...    ...     ...
1            0      0       0,0,1,0,0,0,0,0,0,0,1...
1            0      1       0,0,0,0,0,1,0,1,0,0,0...
...          ...    ...     ...
===========  =====  ======= ===================================


To put your own data into a :py:class:`pandas.DataFrame`, so it can be used by the models in this toolbox create a MultiIndex for example like this::

	import ni
	import pandas as pd
	d = []
	tuples = []
	for con in range(nr_conditions):
		for t in range(nr_trials):
			for c in range(nr_cells):
					spikes = list(ni.model.pointprocess.getBinary(Spike_times_STC.all_SUA[0][0].spike_times[con,t,c].flatten()*1000))
					if spikes != []:
						d.append(spikes)
						tuples.append((con,t,c))
	index = pd.MultiIndex.from_tuples(tuples, names=['Condition','Trial','Cell'])
	data = ni.data.data.Data(pd.DataFrame(d, index = index))

If you only have one trial if several cells or one cell with a few trials, it can be indexed like this:

	from ni.data.data import Data
	import pandas as pd
	
	index = pd.MultiIndex.from_tuples([(0,0,i) for i in range(len(d))], names=['Condition','Cell','Trial'])
	data = Data(pd.DataFrame(d, index = index))

To use the data you can use :py:func:`ni.data.data.Data.filter`::

	only_first_trials = data.filter(0, level='Trial')

	# filter returns a copy of the Data object

	only_the_first_trial = data.filter(0, level='Trial').filter(0, level='Cell').filter(0, level='Condition') 

	only_the_first_trial = data.condition(0).cell(0).trial(0) # condition(), cell() and trial() are shortcuts to filter that set *level* accordingly

	only_some_trials  = data.trial(range(3,15))
	# using slices, ranges or boolean indexing causes the DataFrame to be indexed again from 0 to N, in this case 0:11

Also ix and xs pandas operations can be useful::

	plot(data.data.ix[(0,0,0):(0,3,-1)].transpose().cumsum())
	plot(data.data.xs(0,level='Condition').xs(0,level='Cell').ix[:5].transpose().cumsum())


"""
from ni.tools.project import *
#import ni.model.ip as _ip
#reload(ni.model.ip)
import pandas
import numpy as np
import scipy
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as pl
from ni.tools.html_view import View
from warnings import warn

MILLISECOND_RESOLUTION = 1000


def saveToFile(path,o):
	""" saves a DataFrame-like to a file """
	return pandas.DataFrame(o).to_pickle(path)

def loadFromFile(path):
	""" loads a pandas DataFrame from a file """
	return pandas.read_pickle(path)


def merge(datas,dim,keys = False):
	"""

	merges multiple Data instances into one::

		data = ni.data.data.merge([ni.data.data.Date(f) for f in ['data1.pkl','data2.pkl','data3.pkl']], dim = 'Data File')

	"""
	cells = 0
	trials = 0
	conditions = 0
	time_bins = 0
	if keys == False:
		keys = range(len(datas))
	for m in datas:
		if dim in m.data.index.names:
			m.data.index = m.data.index.droplevel(dim)#.names[m.data.index.names.index(dim)] = "__"+dim
	df = pandas.concat([m.data for m in datas], keys = keys, names = [dim])
	return Data(df)


def matrix_to_dataframe(matrix, dimensions):
	""" conerts a trial x cells matrix into a DataFrame """
	tup = []
	for t1 in range(self.nr_trials):
		for t2 in range(self.nr_cells):
			tup.append((0,t1,t2))
	df = pandas.DataFrame(matrix)
	df.index = pandas.MultiIndex.from_tuples(tup, names=['Condition','Trial','Cell'])
	return df

conversion_factors = {
		'picoseconds': 0.000000000001, 'ps': 0.000000000001, 
		'nanoseconds': 0.000000001, 'ns': 0.000000001, 
		'microseconds': 0.000001, 'mus': 0.000001, 
		'milliseconds': 0.001, 'ms': 0.001, 
		'seconds': 1.0, 's': 1.0,
		'minutes': 60.0, 'min': 60.0, 
		'hours': 60.0*60.0, 'h': 60.0*60.0 
		}
def convert_time(times, from_units, to_units):
	conversion = conversion_factors[from_units]/conversion_factors[to_units]
	return times * conversion

class SpikeContainer:
	"""
	"""
	def __init__(self,filename_or_data=None,units='ms',min_t=0.0,max_t=None,meta=None):
		self.data_format = 'empty'
		self.spike_times = None
		self.spike_containers = None
		self.meta = meta
		self.units = units
		self.min_t = min_t
		self.max_t = max_t
		if type(filename_or_data) == str:
			try:
				self.load_from_csv(filename_or_data,units,min_t=min_t,max_t=max_t)
			except:
				raise
		if type(filename_or_data) == list:
			self.data_format = 'spike_containers'
			self.spike_containers = []
			for f in filename_or_data:
				try:
					sc = SpikeContainer(f,units=units,min_t=min_t,max_t=max_t,meta={'filename':f,'parent':self})
				except:
					raise
				if self.min_t is None or sc.min_t < self.min_t:
					self.min_t = sc.min_t
				if self.max_t is None or sc.max_t > self.max_t:
					self.max_t = sc.max_t
				self.spike_containers.append(sc)
		if type(matrix) == np.ndarray:
			try:
				self.set_spike_times(filename_or_data,units,min_t=min_t,max_t=max_t)
			except:
				raise
	def plot(self,units='ms',y=0,marker='|',min_t=None,max_t=None,**kwargs):
		"""
		Plots the pointprocess as points at line `y`.

		`marker` determines the color and shape of the marker. Default is a vertical line '|'
		"""
		if self.data_format == 'spike_times':
			spike_times = self.get_spike_times(units)
			if spike_times is not None:
				return pl.plot(spike_times,[y]*len(spike_times),marker,**kwargs)
		if self.data_format == 'spike_containers':
			for y_plus,sc in enumerate(self.spike_containers):
				sc.plot(units=units,y=y+y_plus,marker=marker,**kwargs)
			return y+y_plus
	def plot_arr(self,resolution=1.0,units='ms',min_t=None,max_t=None,**kwargs):
		return pl.plot(
					sc.get_spike_array_index(resolution=resolution,units=units,min_t=min_t,max_t=max_t),
					sc.get_spike_array(resolution=resolution,units=units,min_t=min_t,max_t=max_t),**kwargs)
	def set_empty(self):
		self.data_format = 'empty'
		self.spike_times = None
		self.spike_containers = None
		self.min_t = None
		self.max_t = None
	def set_spike_times(self,data,units='ms',min_t=0.0,max_t=None):
		self.spike_times = data
		self.units = units
		self.min_t = min_t
		self.max_t = max_t
		if self.min_t is None or np.min(self.spike_times) < self.min_t:
			self.min_t = np.min(self.spike_times)
		if self.max_t is None or np.max(self.spike_times) < self.max_t:
			self.max_t = np.max(self.spike_times)
		self.data_format = 'spike_times'
	def get_spike_times(self,units='ms'):
		if self.data_format == 'spike_times':
			return convert_time(self.spike_times,from_units=self.units,to_units=units)
		if self.data_format == 'spike_containers':
			return np.array([sc.get_spike_times(units=units) for sc in self.spike_containers])
	def get_spike_array(self,resolution=1.0,units='ms',min_t=None,max_t=None):
		if self.data_format == 'spike_times':
			times = convert_time(self.spike_times,from_units=self.units,to_units=units)
			if min_t is None:
				min_t = convert_time(self.min_t,from_units=self.units,to_units=units)
			if max_t is None:
				max_t = convert_time(self.max_t,from_units=self.units,to_units=units)
			spike_array = np.zeros(int(ceil((max_t-min_t) / resolution)))
			for t in times:
				# assuming one dimensional data for now?
				if min_t is not None and t < min_t:
					continue
				if  max_t is not None and t > max_t:
					continue
				spike_array[int((t-min_t) / resolution)] += 1
			return spike_array
		if self.data_format == 'empty':
			return np.zeros(int(ceil((max_t-min_t) / resolution)))
		if self.data_format == 'spike_containers':
			if min_t is None:
				min_t = convert_time(self.min_t,from_units=self.units,to_units=units)
			if max_t is None:
				max_t = convert_time(self.max_t,from_units=self.units,to_units=units)
			return np.array([sc.get_spike_array(resolution=resolution,units=units,min_t=min_t,max_t=max_t) for sc in self.spike_containers])
	def get_spike_array_index(self,resolution=1.0,units='ms',min_t=None,max_t=None):
		if min_t is None:
			min_t = convert_time(self.min_t,from_units=self.units,to_units=units)
		if max_t is None:
			max_t = convert_time(self.max_t,from_units=self.units,to_units=units)
		return np.arange(min_t,max_t,resolution)
	def load_from_csv(self,filename,units='ms',delimiter=' ',min_t=0.0,max_t=None):
		import csv
		with open(filename, 'rb') as csvfile:
			c = csv.reader(csvfile, delimiter=delimiter)
			floats = [[float(r) for r in l] for l in c]
			if len(floats) <= 0:
				self.set_empty()
				return
			spike_times = np.squeeze(floats)
			if len(spike_times.shape) == 0:
				spike_times = np.array([spike_times])
			self.set_spike_times(spike_times, units=units,min_t=min_t,max_t=max_t)


class Data:
	"""
		Spike data container

		Contains a panda Data Frame with MultiIndex.
		Can save to and load from files.

		The Index contains at least Trial, Cell and Condition and can be extended.


	"""
	def __init__(self,matrix, dimensions = [], key_index="i", resolution=MILLISECOND_RESOLUTION):
		"""
			Can be initialized with a DataFrame, filename or Data instance

			**resolution**

					resolution in bins per second
		"""
		self.desc = ""
		self.type = "binary_array"
		self.nr_cells = 1
		self.nr_trials = 1
		self.nr_conditions = 1
		self.time_bins = 1
		self.data = pandas.DataFrame(np.zeros((1,1)))
		self.resolution = resolution
		if type(matrix) == str:
			matrix = pandas.read_pickle(matrix)
		if isinstance(matrix, Data):
			self = copy(matrix)
			self.__class__ = Data
		if type(matrix) == np.ndarray:
			if dimensions != []:
				if len(matrix.shape) == 2:
					self.data = pandas.DataFrame(matrix)
					if 'Trial' in dimensions:
						self.nr_trials = matrix.shape[dimensions.index('Trial')]
					if 'Cell' in dimensions:
						self.nr_trials = matrix.shape[dimensions.index('Cell')]
					if 'Condition' in dimensions:
						self.nr_conditions = matrix.shape[dimensions.index('Condition')]
					if 'Time' in dimensions:
						self.trial_length = matrix.shape[dimensions.index('Time')]
					tup = []
					for t1 in range(self.nr_trials):
						for t2 in range(self.nr_cells):
							tup.append((0,t1,t2))
					self.data.index = pandas.MultiIndex.from_tuples(tup, names=['Condition','Trial','Cell'])
				elif len(matrix.shape) == 3:
					raise Exception("Not implemented.")
			elif len(matrix.shape) == 3:
				self.data = pandas.DataFrame(matrix.reshape((matrix.shape[0]*matrix.shape[1],matrix.shape[2])))
				tup = []
				self.nr_trials = matrix.shape[0]
				self.nr_cells = matrix.shape[1]
				self.time_bins = matrix.shape[2]
				for t1 in range(self.nr_trials):
					for t2 in range(self.nr_cells):
						tup.append((0,t1,t2))
				self.data.index = pandas.MultiIndex.from_tuples(tup, names=['Condition','Trial','Cell'])
				self.trial_length = int(self.time_bins/self.nr_trials)
			elif len(matrix.shape) == 2:
				self.data = pandas.DataFrame(matrix)
				self.nr_trials = matrix.shape[0]
				self.nr_cells = 1
				self.time_bins = matrix.shape[1]
				tup=[]
				for t1 in range(self.nr_trials):
					tup.append((0,t1,0))
				self.data.index = pandas.MultiIndex.from_tuples(tup, names=['Condition','Trial','Cell'])
				self.trial_length = int(self.time_bins/self.nr_trials)
			elif len(matrix.shape) == 1:
				self.data = pandas.DataFrame(matrix)
				self.nr_trials = 1
				self.nr_cells = 1
				self.time_bins = matrix.shape[0]
				self.trial_length = int(self.time_bins/self.nr_trials)
			else:
				raise Exception("Matrix has incompatible dimensions. Consider using a pandas.DataFrame.")
		elif type(matrix) == pandas.core.frame.DataFrame:
			self.data = matrix.fillna(value=0)
			ind = dict(zip(*[matrix.index.names, range(len(matrix.index.names))]))
			self.nr_conditions = 1
			self.nr_trials = 1
			self.nr_cells = 1
			if type(matrix.index) == pandas.MultiIndex:
				if 'Condition' in ind:
					self.nr_conditions = matrix.index.levshape[ind['Condition']]
				if 'Trial' in ind:
					self.nr_trials = matrix.index.levshape[ind['Trial']]
				if 'Cell' in ind:
					self.nr_cells = matrix.index.levshape[ind['Cell']]
				self.time_bins = matrix.shape[1]
				self.trial_length = self.time_bins#int(self.time_bins/self.nr_trials)
			else:
				if 'Condition' in ind:
					self.nr_conditions = matrix.index.shape[ind['Condition']]
				if 'Trial' in ind:
					self.nr_trials = matrix.index.shape[ind['Trial']]
				if 'Cell' in ind:
					self.nr_cells = matrix.index.shape[ind['Cell']]
				self.time_bins = matrix.shape[1]
				self.trial_length = self.time_bins#int(self.time_bins/self.nr_trials)
		elif type(matrix) == pandas.core.series.Series:
			self.data = matrix.fillna(value=0)
			self.nr_conditions = 1
			self.nr_trials = 1
			self.nr_cells = 1
			self.time_bins = matrix.shape[0]
			self.trial_length = self.time_bins
		elif type(matrix) == list:
			if isinstance(matrix[0], Data):
				self.data = pandas.concat([m.data for m in matrix], keys = range(len(matrix)), names = [key_index])
			if type(self.data.index) == pandas.MultiIndex:
				ind = dict(zip(*[self.data.index.names, range(len(self.data.index.names))]))
				if 'Condition' in ind:
					self.nr_conditions = self.data.index.levshape[ind['Condition']]
				if 'Trial' in ind:
					self.nr_trials = self.data.index.levshape[ind['Trial']]
				if 'Cell' in ind:
					self.nr_cells = self.data.index.levshape[ind['Cell']]
				self.time_bins = self.data.shape[1]
				self.trial_length = self.time_bins#int(self.time_bins/self.nr_trials)
		else:
			self.data = np.array([0])
			self.nr_trials = 0
			self.nr_cells = 0
			self.time_bins = 0
			self.trial_length = self.time_bins
	def cell(self,cells=False):
		"""filters for an array of cells -> see :py:func:`ni.data.data.Data.filter`"""
		if (cells ==[0] or cells == 0) and self.nr_cells == 1:
			return self
		return self.filter(cells,'Cell')
	def condition(self,conditions=False):
		"""filters for an array of conditions -> see :py:func:`ni.data.data.Data.filter`"""
		return self.filter(conditions,'Condition')
	def trial(self,trials=False):
		"""filters for an array of trials -> see :py:func:`ni.data.data.Data.filter`"""
		return self.filter(trials,'Trial')
	def time(self,begin=None,end=None):
		"""gives a copy of the data that contains only a part of the timeseries for all trials,cells and conditions.

		This resets the indices for the timeseries to 0...(end-begin)
		"""
		if begin == None:
			begin = 0
		if end == None or end > len(self.data.columns):
			end = len(self.data.columns)
		data = self.data.iloc[:,begin:end]
		data.columns = range(len(data.columns))
		data.index = pandas.MultiIndex.from_tuples(self.data.index.tolist(), names=self.data.index.names)
		return Data(data)
	def reduce_resolution(self,factor=2):
		if factor == 1:
			return Data(self.data)
		before_spikes = self.data.sum().sum()
		data = pandas.concat([ self.data.iloc[:,int(a*factor):int((a+1)*factor)].max(axis=1) for a in range(int(self.time_bins/factor)) ],axis=1)
		data.columns = range(len(data.columns))
		data.index = pandas.MultiIndex.from_tuples(self.data.index.tolist(), names=self.data.index.names)
		after_spikes = data.sum().sum()
		if before_spikes != after_spikes:
			warn("Lost "+str(int(before_spikes - after_spikes))+" spikes in the process of reducing resolution.", Warning)
		return Data(data)
	def filter(self,array=False,level='Cell'):
		"""filters for arbitrary index levels
			`array` a number, list or numpy array of indizes that are to be filtered
			`level` the level of index that is to be filtered. Default: 'Cell' 
		"""
		if type(array) == bool:
			#dbg("called for nothing")
			return self
		else:
			if type(array) == int or type(array) == float:
				array = [int(array)]
			if type(self.data) == pandas.DataFrame:
				if type(self.data.index) == pandas.Int64Index:
					data = pandas.concat([self.data.ix[i] for i in array],keys=range(len(array)),names=[level])
					return Data(data)
				elif type(self.data.index) == pandas.MultiIndex:
					data = pandas.concat([self.data.xs(i,level=level) for i in array],keys=range(len(array)),names=[level])
					data.index = pandas.MultiIndex.from_tuples(data.index.tolist(), names=data.index.names)
					return Data(data)
				else:
					raise Exception("Unrecognized DataFrame index")
			else:
				raise Exception("Unrecognized Data")
	def firing_rate(self,smooth_width=0,trials=False):
		"""
			computes the firing rate of the data for each cell separately.
		"""
		if type(trials) is bool:
			channels = []
			for cell in range(self.nr_cells):
				n = self.cell(cell)
				d = n.data.sum(0)
				channels.append(scipy.ndimage.gaussian_filter(d,smooth_width)/(self.nr_trials*self.nr_conditions))
			return np.array(channels).transpose()
		else:
			channels = []
			for cell in range(self.nr_cells):
				n = self.cell(cell).trial(trials)
				d = n.data.sum(0)
				channels.append(scipy.ndimage.gaussian_filter(d,smooth_width)/(len(trials)*self.nr_conditions))
			return np.array(channels).transpose()
	def interspike_intervals(self,smooth_width=0,trials=False):
		"""
			computes inter spike intervalls in the data for each cell separately.
		"""
		if type(trials) is bool:
			channels = []
			for cell in range(self.nr_cells):
				n = self.cell(cell)
				d = n.data.sum(0)
				channels.append(scipy.ndimage.gaussian_filter(d,smooth_width))
			return channels
		else:
			channels = []
			for cell in range(self.nr_cells):
				n = self.cell(cell).trial(trials)
				d = n.data.sum(0)
				channels.append(scipy.ndimage.gaussian_filter(d,smooth_width))
			return channels
	def as_series(self):
		"""
		Returns one timeseries, collapsing all indizes.

		The output has dimensions of (N,1) with N being length of one trial x nr_trials x nr_cells x nr_conditions (x additonal indices).

		If cells, conditions or trials should be separated, use :func:`as_list_of_series` instead.
		"""
		if type(self.data) is pandas.core.frame.DataFrame:
			data = self.data.stack()
			data = data.reshape((data.shape[0],1))
			return data
		else:
			data = self.data.reshape((np.prod(self.data.shape),1))
			return data
	def as_list_of_series(self,list_conditions=True,list_cells=True,list_trials=False,list_additional_indizes=True):
		"""
		Returns one timeseries, collapsing only certain indizes (on default only trials). All non collapsed indizes
		"""
		if list_conditions and self.nr_conditions > 1:
			return [self.condition(c).as_list_of_series() for c in range(self.nr_conditions)]
		if list_cells and self.nr_cells > 1:
			return [self.cell(c).as_list_of_series() for c in range(self.nr_cells)]
		if list_trials and self.nr_trials > 1:
			return [self.trial(t).as_list_of_series() for t in range(self.nr_trials)]

		if list_additional_indizes and type(self.data.index) == pandas.MultiIndex:
			ind = dict(zip(*[self.data.index.names, range(len(self.data.index.names))]))
			for n in ind:
				if not n == 'Trial' and not n == 'Condition' and not n == 'Cell':
					if self.data.index.levshape[ind[n]] > 1:
						return [self.filter(a, level=n).as_list_of_series() for a in range(self.data.index.levshape[ind[n]])]
		return self.as_series() 
	def getFlattend(self,all_in_one=True,trials=False):
		"""
		.. deprecated:: 0.1
			Use :func:`as_list_of_series` and :func:`as_series` instead
		Returns one timeseries for all trials.

		The *all_in_one* flag determines whether ``'Cell'`` and ``'Condition'`` should also be collapsed. If set to *False* and the number of Conditions and/or Cells is greater than 1, a list of timeseries will be returned. If both are greater than 1, then a list containing for each condition a list with a time series for each cell.

		"""
		#print "getFlattend"
		if not all_in_one:
			if self.nr_conditions > 1:
				#print "collapsing conditions"
				return [self.condition(c).getFlattend() for c in range(self.nr_conditions)]
			if self.nr_cells > 1:
				#print "collapsing cells"
				return [self.cell(c).getFlattend() for c in range(self.nr_cells)]
		spike_train_all_trial = []
		if not type(trials) is list:
			if type(trials) is bool:
				#print type(self.data.index)
				if type(self.data) is pandas.core.frame.DataFrame:
					data = self.data.stack()
					data = data.reshape((data.shape[0],1))
					#print data.shape
					return data
				else:
					data = self.data.reshape((np.prod(self.data.shape),1))
					#print data.shape
					return data
			else:
				if type(trials) is int and trials <= self.nr_trials:
					trials = range(trials)
					data = self.trial(trials).data.stack()
					data = data.reshape((data.shape[0],1))
					return data
				else: # Whatever data is now
					raise Exception("Unrecognized trial indices")
					trials = range(self.nr_trials)
					data = self.trial(trials).data.stack()
					data = data.reshape((data.shape[0],1))
					return data
		for trial in trials:
			spike_train = []
			if type(self.data.index) is pandas.core.index.Int64Index:
				spike_train = self.data.ix[trial]
			else:
				spike_train = self.data.xs(trial,level='Trial')
			spikes = np.where(spike_train)
			spike_train_all_trial.extend(spike_train)
		spike_train_all_trial_ = np.array(spike_train_all_trial)
		spike_train_all_trial = np.reshape(spike_train_all_trial_,(spike_train_all_trial_.shape[0],1))
		return spike_train_all_trial
	def shape(self,level):
		"""
			Returns the shape of the sepcified level::

				>>> data.shape('Trial')
					100

				>>> data.shape('Cell') == data.nr_cells
					True

		"""
		ind = dict(zip(*[self.data.index.names, range(len(self.data.index.names))]))
		if level in ind:
			return self.data.index.levshape[ind[level]]
	def __str__(self):
		"""
			Returns a string representation of the Data Object.

		"""
		s =  "Spike data: " + str(self.nr_conditions) + " Condition(s) " + str(self.nr_trials) + " Trial(s) of " + str(self.nr_cells) + " Cell(s) in " + str(self.time_bins) + " Time step(s)."
		if type(self.data.index) == pandas.MultiIndex:
			ind = dict(zip(*[self.data.index.names, range(len(self.data.index.names))]))
			additional = [str(n) + " (" +str(self.data.index.levshape[ind[n]]) + ") " for n in ind if not n == 'Trial' and not n == 'Condition' and not n == 'Cell']
			if len(additional) > 0:
				s = s + " Additional indices: " + ", ".join(additional) 
		return s
	def to_pickle(self,path):
		"""
			Saves the DataFrame to a file
		"""
		print "Saving to "+path+"... "
		print self
		print self.data
		print pandas.DataFrame(self.data)
		pandas.DataFrame(self.data).to_pickle(path)
		if not self.desc == "":
			with open(path+".info","w") as f:
				f.write(self.desc)
				f.close()
	def read_pickle(self,path):
		"""
			Loads a DataFrame from a file
		"""
		self.data = pandas.read_pickle(path)
		if os.path.exists(path+".info"):
			with open(path+".info","r") as f:
				self.desc = f.read()
		return self
	def html_view(self):
		view = View()
		data_prefix = ""
		for c in range(self.nr_conditions):
			cond_data = self.condition(c)
			with view.figure(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 0"):
				pl.plot(np.mean(np.array(cond_data.firing_rate()).transpose(),axis=0))
			with view.figure(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 1"):
				pl.plot(np.mean(np.array(cond_data.firing_rate()).transpose(),axis=1))
			with view.figure(data_prefix + "/tabs/Condition " + str(c) + "/Firing Rate/Plot/#4/Mean/tabs/Axis 1 (smoothed)"):
				pl.plot(gaussian_filter(np.mean(np.array(cond_data.firing_rate()).transpose(),axis=1),20))
			for cell in range(cond_data.nr_cells):
				cell_data = cond_data.cell(cell)
				with view.figure(data_prefix + "/tabs/Condition " + str(c) + "/Spikes/tabs/Cell " + str(cell) + "/Plot"):
					pl.imshow(1-cell_data.data,interpolation='nearest',aspect=20,cmap='gray')
		return view
