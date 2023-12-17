import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import struct
import numpy as np
import argparse

dataset_dir = '/media/orlando21/DATA/UPenn/Courses/ESE546PrinciplesOfDeepLearning/final_project/code/MPNet/dataset/'

def main(args):
    # visualize point cloud (obstacles)
    obs = []
    temp=np.fromfile(args.obs_file)
    obs.append(temp)
    obs = np.array(obs).astype(np.float32).reshape(-1,2)
    plt.scatter(obs[:,0], obs[:,1], c='blue')




    # visualize path
    path=np.fromfile(args.path_file)
    # path = np.loadtxt(args.path_file)
    print("Path shape: ", path.shape)
    print("Path:\n", path)
    path = path.reshape(-1, 2)
    path_x = []
    path_y = []
    for i in range(len(path)):
        path_x.append(path[i][0])
        path_y.append(path[i][1])

    plt.plot(path_x, path_y, c='r', marker='o')

    # Generated path
    path=np.fromfile('the_path.dat')
    # path = np.loadtxt(args.path_file)
    print("Path shape: ", path.shape)
    print("Path:\n", path)
    path = path.reshape(-1, 2)
    path_x = []
    path_y = []
    for i in range(len(path)):
        path_x.append(path[i][0])
        path_y.append(path[i][1])

    plt.plot(path_x, path_y, c='g', marker='o')

    plt.show()


parser = argparse.ArgumentParser()
# for training
# parser.add_argument('--obs_file', type=str, default=dataset_dir + 'obs_cloud/obc0.dat',help='obstacle point cloud file')
# parser.add_argument('--path_file', type=str, default=dataset_dir + 'e0/path0.dat',help='path file')
parser.add_argument('--obs_file', type=str, default=dataset_dir + 'obs_cloud/obc0.dat',help='obstacle point cloud file')
parser.add_argument('--path_file', type=str, default=dataset_dir + 'e0/path4001.dat',help='path file')
# parser.add_argument('--path_file', type=str, default='the_path.dat',help='path file')
args = parser.parse_args()
print(args)
main(args)
