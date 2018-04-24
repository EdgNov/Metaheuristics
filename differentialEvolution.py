import sys
import numpy as np
from math import exp
from random import shuffle
import random as rnd
import math
import csv
import matplotlib.pyplot as plt
class DiffEvolution(object):

	def __init__(self, size, _N, _K, _tN):
		self.size = size
		
		self.N 		= _N
		self.K 		= _K
		self.testN 	= _tN
		self.testK  = _K
		self.inputs  	 = []
		self.outputs 	 = []
		self.testInputs  = []
		self.testOutputs = []
		
		self.parents = []
		self.bestCost = 999999;
		
		self.c = 1;
		
	def setInputsOutputs(self, _inputs, _outputs):
		self.inputs  = _inputs
		self.outputs = _outputs

	
	def setTestInputsOutputs(self,  _inputs, _outputs):
		self.testInputs  = _inputs
		self.testOutputs = _outputs
		
		
	def setWeights(self, population):
		# Sets all weights to random values
		for i in range(self.size):
			self.parents.append((population[i],self.fitness(population[i])))
			if self.parents[i][1] < self.bestCost:
				self.bestCost = float(self.parents[i][1])
		return self.bestCost
	
	def fitness(self, M):
		err = 0;
		for i in range(self.N):
			output = self.computeOutput(M,self.inputs[i])
			for j in range(self.K):
				err += (output[j] - self.outputs[i][j]) ** 2
		err /= (self.N  * self.K)	
		return err	
		
	def computeOutput(self, M, input):
		output = M.dot(input)
		
		for i in range(self.K):
			output[i] = 1/(1+exp(-output[i]*5))
		
		return output
		
	
	def crossover(self, dest, orig, k):
		# Take two parents (x and y) and make two children by applying k-point
		# crossover. Positions for crossover are chosen randomly.
		for i in range(self.K):
			for j in range(self.K):
				if rnd.randint(0,100)/100 < k:
					dest[i][j] = orig[i][j]
		return dest		
			
	def mutation(self, M, prob):
			# Mutate (i.e. change) each bit in individual x with given probabipity.
		'''if rnd.randint(0,100)/100 < prob:
			x = rnd.randint(0, self.K**2 - 1)
			y = rnd.randint(0, self.K**2 - 1)
			while x == y:
				y = rnd.randint(0, self.K**2 - 1)
			M[x], M[y] = M[y], M[x]'''	
		for i in range(self.K**2):
			if rnd.randint(0,100)/100 < prob:
				M[i] = 1/(1+exp(-M[i]))
		return M
			
	def calc(self, eras):
		era = 0;
		self.c = 1;
		bestCosts = []
		lastBest = 999;
		forBreak = 0;
		while era < eras:
			era += 1
			i = 0
			while i < self.size:
				a = np.random.randint(0,self.size)
				while a == i:
					a = np.random.randint(0,self.size)
				b = np.random.randint(0,self.size)
				while b == a or b == i:
					b = np.random.randint(0,self.size)
				c = np.random.randint(0,self.size)
				while c == b or c == a or c == i:
					c = np.random.randint(0,self.size)
						 
				V1 = self.parents[i][0];
				V  = self.parents[a][0] + 0.9 * ( self.parents[b][0] - self.parents[c][0])
				
				for x in range(self.K):
					for y in range(self.K):
						if V[x][y] < -1 or V[x][y] > 1:
							V[x][y] = math.sin(V[x][y])
				W1 = self.crossover(V,V1,0.1);
				#W1 = np.matrix(V3);
				#W1.resize(self.K,self.K);
				for x in range(self.K):
					for y in range(self.K):
						if W1[x][y] < -1 or W1[x][y] > 1:
							W1[x][y] = math.sin(W1[x][y])
				f = self.fitness(W1)
				if f < self.parents[i][1]:
					self.parents[i] = (W1, f)
					if f < self.bestCost:
						self.bestCost = f[0]
				i += 1
			if lastBest == self.bestCost:
				forBreak += 1
				if forBreak == 10:
					break;
			else:
				forBreak = 0;
				lastBest = self.bestCost	;
			if era % 10 == 0:
				print(era, self.bestCost)
			self.c *= 1;
			bestCosts.append(float(self.bestCost))
		'''
		plt.semilogx(bestCosts)
		plt.show()
		'''
		self.parents = sorted(self.parents, key = lambda fit: fit[1]);
			
		return bestCosts
			
	def test(self, M):
		err = 0;

		for i in range(self.testN):
			output = self.computeOutput(M,self.testInputs[i])
			for j in range(self.testK):
				err += (output[j] - self.testOutputs[i][j]) ** 2
		err /= (self.testN * self.testK)	
		
		return err

		
if __name__ == "__main__":
	
	per = DiffEvolution(40)
	#per.readFromFile()
	#print(per.parents)
	per.calc(400);
	
	#for M in per.parents
	#	print(per.fitness(M),"\n=======================")
	#print(per.parents)
	
	print(per.computeOutput(per.parents[0][0],per.inputs[1]),"\n===\n",per.outputs[1],"\n===\n",float(per.parents[0][1]))
	
	s = sys.stdin.read(1)
	
	while s != 'n':
		per.calc(100);
		print(per.computeOutput(per.parents[0][0],per.inputs[1]),"\n===\n",per.outputs[1],"\n===\n",per.fitness(per.parents[0][0]))
		s = sys.stdin.read(1)
	
	