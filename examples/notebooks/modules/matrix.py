# 03/01/18
# Chris Self

# python3 compatibility (http://python-future.org/imports.html)
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt

defaults = {'figsize':(8,8),'cmap':'viridis'}

def rmat(matrix,figsize=None,cmap=None):
	"""
	display a real valued matrix
	"""
	if not figsize:
		figsize = defaults['figsize']
	if not cmap:
		cmap = defaults['cmap']

	fig = plt.figure( figsize=figsize )
	plt.imshow(matrix,interpolation='nearest',cmap=cmap)
	plt.colorbar()
	return fig

def cmat(matrix,figsize=None,cmap=None):
	"""
	display a real and imaginary parts of a complex valued matrix side by side
	"""
	if not figsize:
		figsize = defaults['figsize']
	if not cmap:
		cmap = defaults['cmap']

	f, (ax1, ax2) = plt.subplots(1,2,figsize=(2*figsize[0],figsize[1]))
	imreal = ax1.imshow(matrix.real,interpolation='nearest',cmap=cmap)
	f.colorbar(imreal,ax=ax1)
	imimag = ax2.imshow(matrix.imag,interpolation='nearest',cmap=cmap)
	f.colorbar(imimag,ax=ax2)
	return f

def twomat(matrix1,matrix2,figsize=None,cmap=None):
	"""
	display two different real valued matrices side by side
	"""
	if not figsize:
		figsize = defaults['figsize']
	if not cmap:
		cmap = defaults['cmap']

	f, (ax1, ax2) = plt.subplots(1,2,figsize=(2*figsize[0],figsize[1]))
	imreal = ax1.imshow(matrix1,interpolation='nearest',cmap=cmap)
	f.colorbar(imreal,ax=ax1)
	imimag = ax2.imshow(matrix2,interpolation='nearest',cmap=cmap)
	f.colorbar(imimag,ax=ax2)
	return f

def matdiff(matrix1,matrix2,figsize=None,cmap=None):
	"""
	display the difference between two real matrices, alongside this plot this difference
	on a log- colour scale (if diff!=0)
	"""
	if not figsize:
		figsize = defaults['figsize']
	if not cmap:
		cmap = defaults['cmap']

	_matdiff = matrix1-matrix2

	f, (ax1, ax2) = plt.subplots(1,2,figsize=(2*figsize[0],figsize[1]))
	imreal = ax1.imshow(_matdiff,interpolation='nearest',cmap=cmap)
	f.colorbar(imreal,ax=ax1)
	# trying to plot the log-scale diff will fail if the difference is zero everywhere
	if not np.all(_matdiff==np.zeros(_matdiff.shape)):
		imimag = ax2.imshow(np.log10(np.abs(_matdiff)),interpolation='nearest',cmap=cmap)
		f.colorbar(imimag,ax=ax2)
	return f
