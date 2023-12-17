import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import os.path
import random
from tqdm import tqdm


def load_dataset(N=30000,NP=1800):

	obstacles=np.zeros((N,2800),dtype=np.float32)
	for i in tqdm(range(0,N)):
		temp=np.fromfile('/media/orlando21/DATA/UPenn/Courses/ESE546PrinciplesOfDeepLearning/final_project/code/MPNet/dataset/obs_cloud/obc'+str(i)+'.dat')
		temp=temp.reshape(len(temp)//2,2)
		obstacles[i]=temp.flatten()

	
	return 	obstacles	
