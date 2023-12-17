import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import load_test_dataset 
from model import MLP 
from torch.autograd import Variable 
import math
import time
from LSTM_model import LSTM_MLP
from copy import deepcopy

size=5.0

"""
###############################
Load Models and Dataset
###############################
"""

# Load trained model for path generation
mlp = MLP(32, 2) # simple @D
mlp.load_state_dict(torch.load('models/mlp_100_4000_PReLU_ae_dd250.pkl'))

lstm_mlp = LSTM_MLP(28,32,2)
lstm_mlp.load_state_dict(torch.load('models/mlp_final.pkl'))

# Move to the GPU
if torch.cuda.is_available():
	mlp.cuda()
	lstm_mlp.cuda()

# Load test dataset
obc, obstacles, paths, path_lengths= load_test_dataset() 



"""
###############################
Path Generation Helper Methods
###############################
"""

def IsInCollision(x,idx):
	s=np.zeros(2,dtype=np.float32)
	
	s[0]=x.flatten()[0]
	s[1]=x.flatten()[1]
	for i in range(0,7):
		cf=True
		for j in range(0,2):
			if abs(obc[idx][i][j] - s[j]) > size/2.0:
				cf=False
				break
		if cf==True:						
			return True
	return False

def steerTo (start, end, idx):
	start, end = start.flatten(), end.flatten()
	DISCRETIZATION_STEP=0.01
	dists=np.zeros(2,dtype=np.float32)
	for i in range(0,2): 
		dists[i] = end[i] - start[i]

	distTotal = 0.0
	for i in range(0,2): 
		distTotal =distTotal+ dists[i]*dists[i]

	distTotal = math.sqrt(distTotal)
	if distTotal>0:
		incrementTotal = distTotal/DISCRETIZATION_STEP
		for i in range(0,2): 
			dists[i] =dists[i]/incrementTotal



		numSegments = int(math.floor(incrementTotal))

		stateCurr = np.zeros(2,dtype=np.float32)
		for i in range(0,2): 
			stateCurr[i] = start[i]
		for i in range(0,numSegments):

			if IsInCollision(stateCurr,idx):
				return 0

			for j in range(0,2):
				stateCurr[j] = stateCurr[j]+dists[j]


		if IsInCollision(end,idx):
			return 0


	return 1

def feasibility_check(path,idx):
	# Checks the feasibility of entire path including the path edges
	for i in range(0,len(path)-1):
		ind=steerTo(path[i],path[i+1],idx)
		if ind==0:
			return 0
	return 1

def collision_check(path,idx):
	# Checks the feasibility of path nodes only
	for i in range(0,len(path)):
		if IsInCollision(path[i],idx):
			return 0
	return 1

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x, volatile=volatile)

def get_input(i,dataset,targets,seq,bs):
	bi=np.zeros((bs,18),dtype=np.float32)
	bt=np.zeros((bs,2),dtype=np.float32)
	k=0	
	for b in range(i,i+bs):
		bi[k]=dataset[seq[i]].flatten()
		bt[k]=targets[seq[i]].flatten()
		k=k+1
	return torch.from_numpy(bi),torch.from_numpy(bt)

def is_reaching_target(start1,start2):
	s1=np.zeros(2,dtype=np.float32)
	s1[0]=start1[0]
	s1[1]=start1[1]

	s2=np.zeros(2,dtype=np.float32)
	s2[0]=start2[0]
	s2[1]=start2[1]

	for i in range(0,2):
		if abs(s1[i]-s2[i]) > 1.0: 
			return False
	return True

def lvc(path,idx):
	# Lazy vertex contraction 
	for i in range(0,len(path)-1):
		for j in range(len(path)-1,i+1,-1):
			ind=0
			ind=steerTo(path[i],path[j],idx)
			if ind==1:
				pc=[]
				for k in range(0,i+1):
					pc.append(path[k])
				for k in range(j,len(path)):
					pc.append(path[k])

				return lvc(pc,idx)
				
	return path

def re_iterate_path2(p,g,idx,obs, model):
	step=0
	path=[]
	path.append(p[0])
	for i in range(1,len(p)-1):
		if not IsInCollision(p[i],idx):
			path.append(p[i])
	path.append(g)			
	new_path=[]
	for i in range(0,len(path)-1):
		target_reached=False

	 
		st=path[i]
		gl=path[i+1]
		steer=steerTo(st, gl, idx)
		if steer==1:
			new_path.append(st)
			new_path.append(gl)
		else:
			itr=0
			target_reached=False
			while (not target_reached) and itr<50 :
				new_path.append(st)
				itr=itr+1
				ip=torch.cat((obs,st,gl))
				ip=to_var(ip)
				ip = ip.reshape(1,-1)
				print("shape of ip", ip.shape)
				st=model(ip)
				st=st.data.cpu()		
				target_reached=is_reaching_target(st,gl)
			if target_reached==False:
				return 0

	#new_path.append(g)
	return new_path

def replan_path(p,g,idx,obs, model):
	path=[]
	path.append(p[0])
	for i in range(1,len(p)-1):
		if not IsInCollision(p[i],idx):
			path.append(p[i])
	path.append(g)			
	new_path=[]
	for i in range(0,len(path)-1):
		target_reached=False

	 
		st=path[i]
		gl=path[i+1]
		steer=steerTo(st, gl, idx)
		if steer==1:
			new_path.append(st)
			new_path.append(gl)
		else:
			itr=0
			pA=[]
			pA.append(st)
			pB=[]
			pB.append(gl)
			target_reached=0
			tree=0
			while target_reached==0 and itr<50 :
				itr=itr+1
				if tree==0:
					ip1=torch.cat((obs,st.flatten(),gl.flatten()))
					ip1=to_var(ip1)
					ip1 = ip1.reshape(1,-1)
					st=model(ip1)
					st=st.data.cpu()
					pA.append(st.flatten())
					tree=1
				else:
					ip2=torch.cat((obs,gl.flatten(),st.flatten()))
					ip2=to_var(ip2)
					ip2 = ip2.reshape(1,-1)
					gl=model(ip2)
					gl=gl.data.cpu()
					pB.append(gl.flatten())
					tree=0		
				target_reached=steerTo(st, gl, idx)
			if target_reached==0:
				return 0
			else:
				for p1 in range(0,len(pA)):
					new_path.append(pA[p1])
				for p2 in range(len(pB)-1,-1,-1):
					new_path.append(pB[p2])

	return new_path	




"""
###############################
Path Generation
###############################
"""

def generate_path(env_idx, start_pos, goal_pos, model):
	"""
	env_idx: index of the environment. Should be value between 101 and 110
	start: start position of the path. Numpy array of shape (2,) and dtype float32
	goal: goal position of the path. Numpy array of shape (2,) and dtype float32
	"""
	path_is_feasible = False
	planning_time = 0.0

	start=torch.from_numpy(deepcopy(start_pos))
	goal=torch.from_numpy(deepcopy(goal_pos))
	
	obs=obstacles[env_idx]
	obs=torch.from_numpy(obs)

	# Path from start to goal
	path1=[] 
	path1.append(start.flatten())

	# Path from goal to start
	path2=[]
	path2.append(goal.flatten())

	# Full path
	path=[] # stores end2end path by concatenating path1 and path2
	target_reached=0
	tree=0
	step=0	

	tic = time.perf_counter()
	while target_reached==0 and step<80 :
		step=step+1
		if tree==0:
			inp1=torch.cat((obs, start.flatten(), goal.flatten()))
			inp1=to_var(inp1)
			inp1 = inp1.reshape(1,-1)

			start=model(inp1)
			start=start.data.cpu()

			path1.append(start.flatten())
			tree=1
		else:
			inp2=torch.cat((obs, goal.flatten(), start.flatten()))
			inp2=to_var(inp2)
			inp2 = inp2.reshape(1,-1)

			goal=model(inp2)
			goal=goal.data.cpu()

			path2.append(goal.flatten())
			tree=0

		target_reached=steerTo(start, goal, env_idx)

	if target_reached==1:
		# Concatenate path1 and path2
		for p1 in range(0,len(path1)):
			path.append(path1[p1])
		for p2 in range(len(path2)-1,-1,-1):
			path.append(path2[p2])
										
		path=lvc(path, env_idx)
		path_is_feasible=feasibility_check(path, env_idx)
		if path_is_feasible==1:
			toc = time.perf_counter()
			planning_time = toc-tic
		else:
			sp=0
			path_is_feasible=0
			while path_is_feasible==0 and sp<10 and path!=0:
				sp=sp+1
				new_goal=torch.from_numpy(goal_pos)
				path=replan_path(path, new_goal, env_idx, obs, model) #replanning at coarse level
				if path!=0:
					path=lvc(path, env_idx)
					path_is_feasible=feasibility_check(path,env_idx)
		
				if path_is_feasible==1:
					toc = time.perf_counter()
					planning_time=toc-tic

	path = np.array([x.numpy().flatten() for x in path])
	path = path.astype(np.float64)
	return path, planning_time, path_is_feasible	

def save_path(path, name):
	path = path.flatten()
	path.tofile("results/" + name + ".dat")


def main(args):
	# Create model directory
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	
	# Create results directory
	if not os.path.exists('./results/'):
		os.makedirs('./results/')
	
	start= np.array([10., -10.], dtype=np.float32)
	goal = np.array([10., 10.], dtype=np.float32)
	lstm_mlp_path, lstm_plan_time, lstm_feasible = generate_path(10, start, goal, lstm_mlp)
	mlp_path, mlp_plan_time, mlp_feasible = generate_path(10, start, goal, mlp)
	
	print("MLP:")
	print("\tFeasible: ", mlp_feasible)
	print("\tPlanning time: ", mlp_plan_time)
	print("\tPath:\n", mlp_path)
	save_path(mlp_path, "mlp_e10_path0")

	print("LSTM + MLP:")
	print("\tFeasible: ", lstm_feasible)
	print("\tPlanning time: ", lstm_plan_time)
	print("\tPath:\n", lstm_mlp_path)
	save_path(lstm_mlp_path, "lstm_mlp_e10_path0")
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str, default='./models/',help='path for saving trained models')
	parser.add_argument('--no_env', type=int, default=50,help='directory for obstacle images')
	parser.add_argument('--no_motion_paths', type=int,default=2000,help='number of optimal paths in each environment')
	parser.add_argument('--log_step', type=int , default=10,help='step size for prining log info')
	parser.add_argument('--save_step', type=int , default=1000,help='step size for saving trained models')

	# Model parameters
	parser.add_argument('--input_size', type=int , default=68, help='dimension of the input vector')
	parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
	parser.add_argument('--hidden_size', type=int , default=256, help='dimension of lstm hidden states')
	parser.add_argument('--num_layers', type=int , default=4, help='number of layers in lstm')

	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=28)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	args = parser.parse_args()

	print("Arguments:\n", args)

	main(args)


