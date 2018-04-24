import sys
import numpy as np
from math import exp
from random import shuffle
import random as rnd
import math
import csv
import matplotlib.pyplot as plt

import PSO
import differentialEvolution as DE
import genetic as GA
class DataCreate(object):
	
	def __init__(self):
		self.inputs  = []	
		self.outputs = []
		
		self.N = 0
		self.K = 0
		self.testN = 0
		self.testInputs  = []
		self.testOutputs = []
		
		self.population = []
		
##==========================================================================

	def readFromFile(self):

		ifile = open('C:\\Users\\Edgar\\Desktop\\dyplomowa\\inputs.csv')
		reader = csv.reader(ifile, delimiter = ',')
		 
		rownum = 0
		for row in reader:
		# Save header row.
			if not rownum ==0:
				colnum = 0
				if colnum > 116:
					break
				for col in row:
					if colnum > 116:
						break
					self.inputs[colnum].append([float(col)])
					colnum += 1
			else:
				for col in row:
					self.inputs.append([])	
			rownum += 1
		self.N = colnum;
		self.K = rownum-1;
		
		ifile.close()
		
		ifile = open('C:\\Users\\Edgar\\Desktop\\dyplomowa\\outputs.csv')
		reader = csv.reader(ifile, delimiter = ',')
		 
		rownum = 0
		for row in reader:
		# Save header row.
			if not rownum ==0:
				colnum = 0
				for col in row:
					if colnum > 116:
						break
					self.outputs[colnum].append([float(col)])
					colnum += 1
			else:
				for col in row:
					self.outputs.append([])	
			rownum += 1
		self.N = colnum;
		self.K = rownum-1;
		ifile.close()
		
##=======================================================================	
	
	def readTestFile(self):

		ifile = open('C:\\Users\\Edgar\\Desktop\\dyplomowa\\tinputs.csv')
		reader = csv.reader(ifile, delimiter = ',')
		 
		rownum = 0
		for row in reader:
		# Save header row.
			if not rownum ==0:
				colnum = 0
				if colnum > 116:
					break
				for col in row:
					if colnum > 116:
						break
					self.testInputs[colnum].append([float(col)])
					colnum += 1
			else:
				for col in row:
					self.testInputs.append([])	
			rownum += 1
		self.oN = colnum;
		self.oK = rownum-1;
		
		ifile.close()
		
		ifile = open('C:\\Users\\Edgar\\Desktop\\dyplomowa\\toutputs.csv')
		reader = csv.reader(ifile, delimiter = ',')
		 
		rownum = 0
		for row in reader:
		# Save header row.
			if not rownum ==0:
				colnum = 0
				for col in row:
					if colnum > 116:
						break
					self.testOutputs[colnum].append([float(col)])
					colnum += 1
			else:
				for col in row:
					self.testOutputs.append([])	
			rownum += 1
		self.testN = colnum;
		self.K = rownum-1;
		ifile.close()

##=======================================================================

	def generatePopulation(self, size):
		for i in range(size):
			self.population.append( np.random.uniform( low = -1.0, high = 1.0, size = (self.K, self.K) ) )
			


		
##======================= Main section ======================================
		
if __name__ == "__main__":
	
	population = 25
	
	  
	##======================== Data Creation ========================
	
	data = DataCreate()
	data.readFromFile()
	data.readTestFile()
	data.generatePopulation(population)
	
	##==================== Differential Evolution ===================
	
	print("\n\n##==================== Differential Evolution ===================\n\n")
	de = DE.DiffEvolution(population, data.N, data.K, data.testN)
	de.setInputsOutputs(data.inputs, data.outputs)
	de.setTestInputsOutputs(data.testInputs, data.testOutputs)
	maxY = de.setWeights(data.population)
	statsDE = de.calc(30)
	print(de.parents[0][1], de.test(de.parents[0][0]))
	
	plt.subplot(1, 3, 1);
	plt.ylim(0, maxY)
	plt.semilogx(statsDE)
	plt.title("Differential Evolution")
	
	##======================= Genetic Algorithm =====================
	
	print("\n\n##======================= Genetic Algorithm =====================\n\n")
	
	ga = GA.Genetic(population, data.N, data.K, data.testN)
	ga.setInputsOutputs(data.inputs, data.outputs)
	ga.setTestInputsOutputs(data.testInputs, data.testOutputs)
	ga.setWeights(data.population)
	statsGA = ga.calc(20)
	print(ga.parents[0][1], ga.test(ga.parents[0][0]))
	plt.subplot(1, 3, 2);
	plt.ylim(0, maxY)
	plt.semilogx(statsGA)
	plt.title("Genetic Algorithm")
	
	##================== Particle Swarm Optimization ================
	
	print("\n\n##================== Particle Swarm Optimization ================\n\n")
	pso = PSO.PSO(population, data.N, data.K, data.testN)
	pso.setInputsOutputs(data.inputs, data.outputs)
	pso.resetPopulation(data.population)
	pso.setTestInputsOutputs(data.testInputs, data.testOutputs)
	statsPSO = pso.calc(50)
	print("Train: ", pso.globalBest.cost,"\nTest: ", pso.test(pso.globalBest.position))
	
	
	
		
	plt.subplot(1, 3, 3);
	plt.ylim(0, maxY)
	plt.semilogx(statsPSO)
	plt.title("\n\nParticle Swarm Optimization")
	
	plt.show()
	