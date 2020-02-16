from random import random
from copy import deepcopy

def f(net):
	if net >= 0.5:
		return 1
	return 0

class Perceptron(object):
	X = None
	Y = None
	weights = None
	lin = None
	col = None 
	def __init__(self,dataset,eta,threshold):
		self.X = []
		self.lin = len(dataset)
		self.col = len(dataset[0])
		# validar se todas linhas tem co colunas
		for i in range(self.lin):
			self.X.append([])
			for j in range(self.col-1):
				self.X[i].append(dataset[i][j])
				
		self.Y = [dataset[i][self.col-1] for i in range(self.lin)]
	
		self.weights = [(random()-0.5) for i in range(self.col)]
		
		if threshold is None:
			threshold = 1e-3
		if eta is None:
			eta = 0.1
			
		sqerror = 2*threshold
		
		#treinamento da rede perceptron
		while sqerror > threshold:
			sqerror = 0
			for i in range(self.lin):
				x = self.X[i]
				y = self.Y[i]
				c = [x[i] for i in range(self.col-1)]
				c.append(1)
				
				# ~ print(len(x), len(c))
				
				net = 0
				for idx in range(self.col):
					net = net + c[idx] * self.weights[idx]
				
				yo = f(net)
				error = y - yo
				sqerror = sqerror + error*error
				
				#somatorio das derivadas
				dE2 = 0
				for idx in range(self.col):
					dE2 = dE2 + 2*(error) * -c[idx]
					
				for idx in range(self.col):
					self.weights[idx] = self.weights[idx] - eta * dE2

	def test(self,x):
		net = 0
		for i in range(self.col-1):
			net = net + self.weights[i]*x[i]
		return f(net) 
		
if __name__ == '__main__':
	dataset = [[0,0,0],[1,0,1],[0,1,1],[1,1,1]]
	rede = Perceptron(dataset,0.01,0.00001)
	print("ok!")
	while True:
		a,b = map(int,input().split())
		aux = [a,b]
		print(aux," = ",rede.test(aux))
	
	
