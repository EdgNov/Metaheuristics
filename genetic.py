import sys
import numpy as np
from math import exp
from random import shuffle
import random as rnd
import math
import csv
import matplotlib.pyplot as plt
class Genetic(object):

	def __init__(self, size):
		self.size = size
		#self.N = 4
		#self.K = 3
		self.parents = []
		self.inputs = [];
		self.outputs = [];
		self.readFromFile();
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
		self.children = [];
	'''
	def reset_weights(self):
		# Sets all weights to random values
		for i in range(self.size):
			p = np.random.rand(3,3) * 2 - 1  
			self.parents.append((p,self.fitness(p)))
		self.children = [];'''
	
	def fitness(self, M):
		err = 0;
		for i in range(self.N):
			output = self.computeOutput(M,self.inputs[i])
			for j in range(self.K):
				err += (output[j] - self.outputs[i][j]) ** 2
		err /= (self.N * self.K)	
		return err	
		
	def computeOutput(self, M, input):
		output = M.dot(input)
		
		for i in range(self.K):
			output[i] = 1/(1+exp(-output[i]*5))
		
		return output
		
	
	def crossover(self, x, y, k):
		# Take two parents (x and y) and make two children by applying k-point
		# crossover. Positions for crossover are chosen randomly.
		x_new = []
		y_new = []
		i = 0;
		while i < self.K**2 - 2:
			x_new.append(x[i])
			x_new.append(y[i+1]);
			y_new.append(y[i]);
			y_new.append(x[i+1]);
			i += 2
		x_new.append(x[8])
		y_new.append(y[8]);	
		x_new = self.mutation(x_new, 0.25)
		y_new = self.mutation(y_new, 0.25)
		return (x_new, y_new)		
	
	def roulette(pop):
		totFit = 0;
		for i in pop:
			totFit += self.parents[i][1];
		
		spin = rnd.randint(0,totFit);
		
		tmpFit = 0;
		
		for i in pop:
			tmpFit += self.parents[i][1];
			if tmpFit >= spin:
				return i;
		return None;

		
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
	def randoms(self, size):
		self.ranoms = []
		for i in range(int(size)):
			p = np.random.uniform( low = -1.0, high = 1.0, size = (self.K, self.K) )
			self.ranoms.append((p,self.fitness(p)))
	def calc(self, eras):
		era = 0;
		self.reset_weights()
		bestCosts = [];
		lastBest = 999;
		forBreak = 0;
		while era < eras:
			era += 1
			self.parents = sorted(self.parents, key = lambda fit: fit[1]);
			bestCosts.append(self.parents[0][1])
			self.randoms(self.size % int(self.size * 3 / 4))
			self.parents = self.parents[:int(self.size * 3 / 4)] + self.ranoms[:];
			self.parents = sorted(self.parents, key = lambda fit: fit[1]);
			if lastBest == self.parents[0][1]:
				forBreak += 1
				if forBreak == 10:
					break;
			else:
				forBreak = 0;
				lastBest = self.parents[0][1];
			self.children = []
			i = 0
			while i < self.size - 1:
				V1 = np.asarray(self.parents[i][0]).ravel();
				V2 = np.asarray(self.parents[i+1][0]).ravel();
				V3, V4 = self.crossover(V1.tolist(),V2.tolist(),7);
				W1 = np.matrix(V3);
				W2 = np.matrix(V4);
				W1.resize(3,3);
				W2.resize(3,3);
				self.children.append((W1,self.fitness(W1)))
				self.children.append((W2,self.fitness(W2)))
				i += 2
			self.children = sorted(self.children, key = lambda fit: fit[1]);
			self.parents = self.parents[:int(self.size/2)] + self.children[:int(self.size/2)];
			if era % 10 == 0:
				print(era) 
		plt.semilogx(bestCosts)
		plt.show()	
			
			
		
if __name__ == "__main__":
	
	per = Genetic(20) 
	per.calc(300);
	
	#for M in per.parents:
	#	print(per.fitness(M),"\n=======================")
	print(per.parents)
	print(per.computeOutput(per.parents[0][0],per.inputs[1]),"\n===\n",per.outputs[1],per.parents[0][1])
	
	
	