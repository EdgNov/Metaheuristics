import sys
import numpy as np
from math import exp
from random import shuffle
import random as rnd
import math
import csv
import matplotlib.pyplot as plt
##=========================================

class best(object):
	def __init__(self, position, cost):
		self.position 	= position
		self.cost		= cost

##=========================================


class particle(object):
	def __init__(self, K):
		self.position 	= np.random.uniform( low = -1.0, high = 1.0, size = (K,K) )
		self.velocity 	= np.zeros( (K,K) )
		self.cost		= 0
		self.best 		= best(self.position, self.cost)
	

##=========================================	
	
class PSO(object):

	def __init__(self, size):
		self.kappa = 1
		self.phi1 = 2.05
		self.phi2 = 2.05
		self.phi  = self.phi1 + self.phi2
		self.chi = 2 * self.kappa/abs(2-self.phi - math.sqrt(self.phi**2-4*self.phi))
		
		
		self.size = size
		self.N = 0
		self.K = 0
		self.oN = 0
		self.oK = 0
		self.w = self.chi
		self.c1 = self.chi * self.phi1;
		self.c2 = self.chi * self.phi2;
		self.inputs = []
		self.outputs = []
		self.testInputs = []
		self.testOutputs = []
		self.particles = []
		self.readFromFile()
		self.globalBest = best(None, 99999)
		self.reset_population()
		
	def readFromFile(self):

		ifile = open('C:\\Users\\Edgar\\Desktop\\dyplomowa\\inputs.csv')
		reader = csv.reader(ifile, delimiter = ',')
		 
		rownum = 0
		for row in reader:
		# Save header row.
			if not rownum ==0:
				colnum = 0
				if colnum > 1000:
					break
				for col in row:
					if colnum > 1000:
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
					if colnum > 1000:
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
	
	def readOutputFile(self):

		ifile = open('C:\\Users\\Edgar\\Desktop\\dyplomowa\\tinputs.csv')
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
					if colnum > 77:
						break
					self.testOutputs[colnum].append([float(col)])
					colnum += 1
			else:
				for col in row:
					self.testOutputs.append([])	
			rownum += 1
		self.N = colnum;
		self.K = rownum-1;
		ifile.close()
	
	def reset_population(self):
		# Sets all weights to random values
		for i in range(self.size):
			p 			= particle(self.K);
			p.cost 		= self.fitness(p.position);
			p.best.cost = p.cost
			self.particles.append(p)
			
			if p.best.cost < self.globalBest.cost:
				self.globalBest = p.best
	def fitness(self, M):
		err = 0;
		
		for i in range(self.N):
			output = self.computeOutput(M,self.inputs[i])
			for j in range(self.K):
				err += (output[j] - self.outputs[i][j]) ** 2
		err /= (self.N * self.K)	
		return err	
	
	def Ofitness(self, M):
		err = 0;
		self.readOutputFile()
		#print(self.testInputs)
		for i in range(self.oN):
			output = self.computeOutput(M,self.testInputs[i])
			print(output, "---",self.testOutputs[i])
			for j in range(self.oK):
				err += (output[j] - self.testOutputs[i][j]) ** 2
		err /= (self.oN * self.oK)	
		return err

		
	def computeOutput(self, M, input):
		output = M.dot(input)
		
		for i in range(self.K):
			try:
				output[i] = 1/(1+exp(-output[i]))
			except ArithmeticError:
				output[i] = 0.00000001
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
		bestCosts = []
		
		while era < eras:
			era += 1
			r1 = np.random.uniform( size = (self.K, self.K) )
			r2 = np.random.uniform( size = (self.K, self.K) )
			
			for i in range(self.size):
				self.particles[i].velocity = self.w * self.particles[i].velocity \
												+ self.c1 * r1 * (self.particles[i].best.position - self.particles[i].position) \
												+ self.c2 * r2 * (self.globalBest.position - self.particles[i].position)
				
				
				self.particles[i].velocity = np.maximum(self.particles[i].velocity, -0.4)
				self.particles[i].velocity = np.minimum(self.particles[i].velocity,  0.4)
				
				self.particles[i].position = self.particles[i].position + self.particles[i].velocity
				
				#self.particles[i].position = np.maximum(self.particles[i].position, -1.0)
				#self.particles[i].position = np.minimum(self.particles[i].position,  1.0)
				
				self.particles[i].cost = self.fitness(self.particles[i].position)
				
				
				if self.particles[i].cost < self.particles[i].best.cost:
					self.particles[i].best.position = self.particles[i].position
					self.particles[i].best.cost		= self.particles[i].cost
					
				
				if self.particles[i].best.cost < self.globalBest.cost:
					self.globalBest = self.particles[i].best
					
			#self.w *= 0.99
			bestCosts.append(self.globalBest.cost)
			if era % 10 == 0:
				print(era, self.globalBest.cost)
		plt.semilogx(bestCosts)
		plt.show()
if __name__ == "__main__":
	
	per = PSO(20)
	#per.readFromFile()
	per.calc(1000)
	print(per.computeOutput(per.globalBest.position,per.inputs[1]),"\n===\n",per.outputs[1],"\n===\n",per.globalBest.cost)
	
	print(per.Ofitness(per.globalBest.position))
	#for p in per.particles:
	#	print(p.position,"\n")
	#print("=",per.globalBest.cost)
	#print(per.particles[0].position * per.particles[1].position)
	#per.calc(100);
	
	#for M in per.parents
	#	print(per.fitness(M),"\n=======================")
	#print(per.parents)
	
	#print(per.computeOutput(per.parents[0][0],per.inputs[1]),"\n===\n",per.outputs[1],"\n===\n",per.parents[0][1])
	
	#s = sys.stdin.read(1)
	
	#while s != 'n':
	#	per.calc(100);
	#	print(per.computeOutput(per.parents[0][0],per.inputs[1]),"\n===\n",per.outputs[1],"\n===\n",per.fitness(per.parents[0][0]))
	#	s = sys.stdin.read(1)
	
	