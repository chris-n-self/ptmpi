#
# 24/04/2016
# Chris Self
#
import sys
import json
from mpi4py import MPI
from contextlib import contextmanager
# use jsci, CT's enhanced json coding, for numpy objs
from jsci import Coding as jscicoding

class filehandler(object):

	def __init__(self,mpi_comm_env,filename='ptmpi',label='T',amode=MPI.MODE_WRONLY|MPI.MODE_CREATE):
		""" """
		self.mpi_comm = mpi_comm_env
		self.fbasename = filename
		self.flabel = label
		self.amode = amode
		self.files = []
		# flag used to track if a comma is needed before the next element
		self.in_array = False
		self.array_between_elements = False

	def __enter__(self):
		""" """
		num_proc = self.mpi_comm.Get_size()
		for rank in range(num_proc):
			self.files.append( MPI.File.Open(self.mpi_comm, self.fbasename+'_'+self.flabel+str(rank)+'.json', self.amode) )
		self.mpi_comm.Barrier()
		return self

	def __exit__(self,*args):
		""" """
		self.mpi_comm.Barrier()
		for f in self.files:
			f.Close()

	def enter_array(self):
		""" """
		self.in_array = True
		if self.mpi_comm.Get_rank()==0:
			for f in self.files:
				f.Write_shared('[\n')
		self.mpi_comm.Barrier()

	def exit_array(self):
		""" """
		self.mpi_comm.Barrier()
		if self.mpi_comm.Get_rank()==0:
			for f in self.files:
				f.Write_shared('\n]')
		self.in_array = False

	@contextmanager
	def wrap_array(self):
		""" """
		self.enter_array()
		try:
			yield
		finally:
			self.exit_array()

	def dump(self,fileindex,obj):
		""" """
		json_string = ''
		if self.array_between_elements:
			json_string = ',\n'
		json_string = json_string + json.dumps(obj,cls=jscicoding.NumericEncoder,indent=2)
		self.files[fileindex].Write_shared(json_string)
		self.array_between_elements = (self.in_array and True) # equiv to just self.in_array, but meaning is clearer
