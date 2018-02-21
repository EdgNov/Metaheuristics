import sys
import numpy as np
from math import exp
from random import shuffle
import random as rnd
import math
import csv
import matplotlib.pyplot as plt
class DiffEvolution(object):

	def __init__(self, size):
		self.size = size
		self.N = 0
		self.K = 0
		self.inputs  = []
		self.outputs = []
		self.parents = []
		self.bestCost = 999999;
		self.readFromFile()
		self.reset_weights()
		self.c = 1;
		
	def readFromFile(self):

		ifile = open('C:\\Users\\Edgar\\Desktop\\dyplomowa\\inputs.csv')
		reader = csv.reader(ifile, delimiter = ',')
		 
		rownum = 0
		for row in reader:
		# Save header row.
			if not rownum ==0:
				colnum = 0
				if colnum > 77:
					break
				for col in row:
					if colnum > 77:
						break
					self.inputs[colnum].append([float(col)])
					colnum += 1
			else:
				for col in row:
					self.inputs.append([])	
			rownum += 1
		self.N = colnum;
		self.K = rownum - 1;
		
		ifile.close()
		
		ifile = open('C:\\Users\\Edgar\\Desktop\\dyplomowa\\outputs.csv')
		reader = csv.reader(ifile, delimiter = ',')
		 
		rownum = 0
		for row in reader:
		# Save header row.
			if not rownum ==0:
				colnum = 0
				for col in row:
					if colnum > 77:
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
	def reset_weights(self):
		# Sets all weights to random values
		for i in range(self.size):
			p = np.random.uniform( low = -1.0, high = 1.0, size = (self.K, self.K) ) 
			self.parents.append((p,self.fitness(p)))
			if self.parents[i][1] < self.bestCost:
				self.bestCost = float(self.parents[i][1])
	
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
		
	
	def crossover(self, x, y, k):
		# Take two parents (x and y) and make two children by applying k-point
		# crossover. Positions for crossover are chosen randomly.
		for i in range(self.K**2):
			if rnd.randint(0,100)/100 < k:
				x[i] = y[i]
		return x		
			
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
						 
				V1 = np.asarray(self.parents[i][0]).ravel();
				V  = np.asarray(self.parents[a][0] + 0.5 * ( self.parents[b][0] - self.parents[c][0])).ravel()
				V3 = self.crossover(V.tolist(),V1.tolist(),0.6);
				W1 = np.matrix(V3);
				W1.resize(self.K,self.K);
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
		
		plt.semilogx(bestCosts)
		plt.show()
		
		self.parents = sorted(self.parents, key = lambda fit: fit[1]);
			
			
			
		
if __name__ == "__main__":
	
	per = DiffEvolution(10)
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
	
	