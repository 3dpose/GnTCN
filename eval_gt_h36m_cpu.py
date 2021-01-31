from models import networkgcn, networktcn 
import torch 
import numpy as np 
from TorchSUL import Model as M 
import datareader 
from tqdm import tqdm
import torch.nn.functional as F 

bone_pairs = [[8,9],[9,10], [8,14],[14,15],[15,16], [8,11],[12,13],[11,12], [8,7],[7,0], [4,5],[5,6],[0,4], [0,1],[1,2],[2,3]]
bone_matrix = np.zeros([16,17], dtype=np.float32)
for i, pair in enumerate(bone_pairs):
	bone_matrix[i, pair[0]] = -1
	bone_matrix[i, pair[1]] = 1
bone_matrix_inv = np.linalg.pinv(bone_matrix)
bone_matrix_inv = torch.from_numpy(bone_matrix_inv)
bone_matrix = torch.from_numpy(bone_matrix)

bsize = 32
seq_len = 243

netgcn = networkgcn.TransNet(256, 17)
nettcn = networktcn.Refine2dNet(17, seq_len)

# initialize the network with dumb input 
x_dumb = torch.zeros(2,17,2)
affb = torch.ones(2,16,16) / 16
affpts = torch.ones(2,17,17) / 17
netgcn(x_dumb, affpts, affb, bone_matrix, bone_matrix_inv)
x_dumb = torch.zeros(2,243, 17*3)
nettcn(x_dumb)

# load networks 
M.Saver(netgcn).restore('./ckpts/model_gcn/')
M.Saver(nettcn).restore('./ckpts/model_tcn/')

# push to gpu 
netgcn.eval()
nettcn.eval()

# get loader 
dataset = datareader.PtsData(seq_len)

# start testing 
sample_num = 0
loss_total = 0
for i in tqdm(range(len(dataset))):
	p2d,p3d = dataset[i]
	bsize = p2d.shape[0]
	affb = torch.ones(bsize,16,16) / 16
	affpts = torch.ones(bsize,17,17) / 17

	with torch.no_grad():
		pred = netgcn(p2d, affpts, affb, bone_matrix, bone_matrix_inv)
		pred = pred.unsqueeze(0).unsqueeze(0)
		pred = F.pad(pred, (0,0,0,0,seq_len//2, seq_len//2), mode='replicate')
		pred = pred.squeeze()
		pred = nettcn.evaluate(pred)
		loss = torch.sqrt(torch.pow(pred - p3d, 2).sum(dim=-1)) # [N, 17]
		loss = loss.mean(dim=1).sum()
		loss_total = loss_total + loss
		sample_num = sample_num + bsize

print('MPJPE: %.4f'%(loss_total / sample_num))
