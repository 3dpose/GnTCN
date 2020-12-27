import numpy as np 
import pickle 
from multiprocessing.pool import ThreadPool
import torch

bone_pairs = [[8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[12,13],[11,12], [8,7],[7,0], [4,5],[5,6],[0,4], [0,1],[1,2],[2,3]]
bone_matrix = np.zeros([16,17], dtype=np.float32)
for i, pair in enumerate(bone_pairs):
	bone_matrix[i, pair[0]] = -1
	bone_matrix[i, pair[1]] = 1
bone_matrix_inv = np.linalg.pinv(bone_matrix)

class PtsData():
	def __init__(self, seq_len):
		self.seq_len = seq_len
		self.half_seq_len = seq_len // 2 
		self.data = pickle.load(open('points_eval.pkl' , 'rb'))

	def __getitem__(self, i):
		seq2d, seq3d = self.data[i]

		res2d = seq2d / 1000
		gt3d = ( seq3d - seq3d[:,:1] ) / 1000

		return torch.from_numpy(np.float32(res2d)), torch.from_numpy(np.float32(gt3d))

	def __len__(self):
		return len(self.data)

	def reset(self):
		pass

if __name__=='__main__':
	dataset = PtsData(243)
	for i in range(len(dataset)):
		pts2d, pts3d = dataset[i]
		print(pts2d.shape, pts3d.shape)
		if i==30:
			break
