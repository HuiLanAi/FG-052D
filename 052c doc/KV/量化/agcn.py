import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import os
import torch.nn.utils.prune as torch_prune
import copy

prune_switch = True
fc_prune_switch = False
conv9x1_prune_switch = True
use_ck = False
forward_cnt = 0


def import_class(name):
	components = name.split('.')
	mod = __import__(components[0])
	for comp in components[1:]:
		mod = getattr(mod, comp)
	return mod


def conv_branch_init(conv, branches):
	weight = conv.weight
	n = weight.size(0)
	k1 = weight.size(1)
	k2 = weight.size(2)
	nn.init.normal(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
	nn.init.constant(conv.bias, 0)


def conv_init(conv):
	nn.init.kaiming_normal(conv.weight, mode='fan_out')
	nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
	nn.init.constant(bn.weight, scale)
	nn.init.constant(bn.bias, 0)



# def quantize(old_tensor, inte_w, deci_w):
#     shift = 2 ** (int)(deci_w)
#     upper = 2 ** (int)(deci_w + inte_w)
#     with torch.no_grad():
# 		q_tensor = old_tensor
# 		q_tensor *= shift
# 		q_tensor = q_tensor.int()
# 		q_tensor[q_tensor > upper] = upper
# 		q_tensor[q_tensor < -upper] = -upper
# 		q_tensor = q_tensor.float()
# 		q_tensor /= shift
#     return q_tensor


def quantize(old_tensor, inte_w, deci_w):
	shift = 2 ** (int)(deci_w)
	upper = 2 ** (int)(deci_w + inte_w)
	with torch.no_grad():
		q_tensor = old_tensor
		q_tensor = q_tensor * shift
		q_tensor = q_tensor.int()
		q_tensor[q_tensor > upper] = upper
		q_tensor[q_tensor < -upper] = -upper
		q_tensor = q_tensor.float()
		q_tensor /= shift
	return q_tensor




def load_drop_list(i):
	data = []
	file_prefix = "/home/winter/2s-AGCN/model/drop/"
	file_name = file_prefix + "L" + str(i) + "_drop.txt"
	f = open(file_name, "r")
	for num in f.readlines():
		data.append((int)(num))
	f.close()
	return data

def load_fc_drop_list():
	data = []
	file_name = "/home/winter/2s-AGCN/model/drop/fc_drop.txt"
	f = open(file_name, "r")
	for num in f.readlines():
		data.append((int)(num))
	f.close()
	return data



def channel_drop_prune(weight, serial):
	zero_size = 0
	total_size = 0
	if(serial == 0):
		return weight
	else:
		with torch.no_grad():
			weight[:, load_drop_list(serial), :, :] = 0

		pruned_weight = weight
		return pruned_weight


def fc_channel_drop_prune(weight):
	with torch.no_grad():
		weight[:, load_fc_drop_list()] = 0
	
	pruned_weight = weight
	return pruned_weight


# tsp2
# sample_scheme_list = [
#     [3, 5, 7],
#     [1, 4],
#     [2, 8],
#     [0, 6],
#     [2, 4, 8],
#     [3, 6],
#     [0, 5],
#     [1, 7]
# ]

# tsp50-0
sample_scheme_list = [
    [0, 2, 4, 6],
    [1, 3, 5, 7],
    [2, 4, 6, 8],
    [3, 5, 7, 0],
    [4, 6, 8, 1],
    [5, 7, 0, 2],
    [6, 8, 1, 3],
    [7, 0, 2, 4]
]


def gen_cavity():
    cavity_list = []
    for i in range (8):
        tmp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        for j in range(len(sample_scheme_list[i])):
            tmp[sample_scheme_list[i][j]] = 100
        
        while (100 in tmp):
            index = tmp.index(100)
            tmp.pop(index)

        cavity_list.append(tmp)
    return cavity_list



def count_sparsity(fea):
	a, b, c, d = fea.shape
	total = a*b*c*d
	fea[torch.abs(fea) != 0] = 1
	none_zero = torch.sum(fea).item()
	return total, none_zero



class unit_tcn(nn.Module):
	def __init__(self, in_channels, out_channels, serial, kernel_size=9, stride=1):
		super(unit_tcn, self).__init__()
		self.pad = int((kernel_size - 1) / 2)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(self.pad, 0),
							stride=(stride, 1))
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		self.stride = stride
		self.serial = serial
		conv_init(self.conv)
		bn_init(self.bn, 1)


	def forward(self, x):
		if(conv9x1_prune_switch):
			cavity_list = gen_cavity()

			o_c, in_c, k, p = self.conv.weight.shape
			if(k > 1 and (self.serial in [0, 1, 2, 3, 4, 5, 6, 7, 8])):
				# tsp method prune for conv9x1
				with torch.no_grad():
					mask = torch.ones(o_c, in_c, k, p).cuda()
					for j in range (0, 8):
						for i in range (j, in_c, 8):
							mask[:, i, cavity_list[j], :] = 0


				# validate dcp method prune for conv9x1
				with torch.no_grad():
					# mask = torch.ones(o_c, in_c, k, p).cuda()
					mask[load_drop_list(self.serial + 1), :, :, :] = 0

					# tsp_mask = mask
					# keep_index = []
					# for i in range(o_c):
					# 	if(i not in load_drop_list(self.serial + 1)):
					# 		keep_index.append(i)
					# for j in range (0, 8):
					# 	for i in range (j, in_c, 8):
					# 		for k in keep_index:
					# 			tsp_mask[k, i, cavity_list[j], :] = 0

				
				torch_prune.custom_from_mask(self.conv, name = "weight", mask = mask)
				torch_prune.remove(self.conv, name = "weight")
				

				self.conv.weight.data = quantize(self.conv.weight.data, 5, 10)
		
		x = self.conv(x)

		x = self.bn(x)
		return x
	

class unit_gcn(nn.Module):
	def __init__(self, in_channels, out_channels, A, serial, coff_embedding=4, num_subset=3):
		super(unit_gcn, self).__init__()
		inter_channels = out_channels // coff_embedding
		self.inter_c = inter_channels
		# PA相当于B
		self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
		nn.init.constant(self.PA, 1e-6)
		self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
		self.num_subset = num_subset

		self.conv_a = nn.ModuleList()
		self.conv_b = nn.ModuleList()
		self.conv_d = nn.ModuleList()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.serial = serial

		for i in range(self.num_subset):
			self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
			self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
			self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
		
		if in_channels != out_channels:
			self.down = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 1),
				nn.BatchNorm2d(out_channels)
			)
		else:
			self.down = lambda x: x

		self.bn = nn.BatchNorm2d(out_channels)
		self.soft = nn.Softmax(-2)
		self.relu = nn.ReLU()

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				conv_init(m)
			elif isinstance(m, nn.BatchNorm2d):
				bn_init(m, 1)
		bn_init(self.bn, 1e-6)
		for i in range(self.num_subset):
			conv_branch_init(self.conv_d[i], self.num_subset)


	def forward(self, x):
		N, C, T, V = x.size()

		A = self.A.cuda(x.get_device())
		A = A + self.PA
		A = quantize(A, 5, 10)

		y = None

		if(prune_switch and not use_ck and self.serial > 0):
			for i in range (self.num_subset):
				# print("cnmcnmcnmcnmc")
				# with torch.no_grad():
				# 	tmp = self.conv_d[i].weight
				# 	a, b, c, d = tmp.shape
				# 	tmp[tmp!=0] = 1
				# 	print(a*b*c*d - torch.sum(tmp).cpu().numpy())

				with torch.no_grad():
					a, b, c, d = self.conv_d[i].weight.shape
					mask = torch.ones(a, b, c, d).cuda()
					mask[:, load_drop_list(self.serial), :, :] = 0

				torch_prune.custom_from_mask(self.conv_d[i], name = "weight", mask = mask)
				torch_prune.remove(self.conv_d[i], name = "weight")
				self.conv_d[i].weight.data = quantize(self.conv_d[i].weight.data, 5, 10)


			
				# with torch.no_grad():
				# 	tmp = self.conv_d[i].weight
				# 	a, b, c, d = tmp.shape
				# 	tmp[tmp!=0] = 1
				# 	print(a*b*c*d - torch.sum(tmp).cpu().numpy())
				# 	print(a*b*c*d)
				# 	debug = input()

		for i in range(self.num_subset):
			# 2, 900, 25
			if use_ck:
				# print(A1.size())
				# conv_a[0](x).size() = 2, 16, 300, 25
				# 2, 25, 4800
				A1 = F.conv2d(x, max_seg_prune(self.conv_a[i].weight), self.conv_a[i].bias)
				A1 = A1.permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
				# print(A2.size())
				# 2, 4800, 25
				A2 = F.conv2d(x, max_seg_prune(self.conv_b[i].weight), self.conv_b[i].bias)
				A2 = A2.view(N, self.inter_c * T, V)
				# 计算C
				# matmul(A1, A2).size() = 2, 25, 25
				# A1.size() = 2, 25, 25
				# 对倒数第2维进行softmax，归一到0~1的范围中
				A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V

				# A1 = A + B + C
				# A1.size() = 2, 25, 25
				A2 = x.view(N, C * T, V)
				A1 = A1 + A[i]

				z = F.conv2d(torch.matmul(A2, A1).view(N, C, T, V), 
						max_seg_prune(self.conv_d[i].weight), self.conv_d[i].bias)
			
			else:
				A2 = x.view(N, C * T, V)
				try:
					# test batch = 72
					A1 = torch.zeros(144, 25, 25).cuda()

					# train batch = 36 / test batch = 80
					# A1 = torch.zeros(160, 25, 25).cuda()

					A1 = A1 + A[i]
					z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
				except RuntimeError:
					try:
						# train batch = 30 / test batch = 30
						# A1 = torch.zeros(60, 25, 25).cuda()

						# train batch = 36 / test batch = 80
						# A1 = torch.zeros(72, 25, 25).cuda()

						# test batch = 72
						A1 = torch.zeros(4, 25, 25).cuda()

						A1 = A1 + A[i]
						z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
					except RuntimeError:
						# # train batch = 30 / test batch = 30
						# A1 = torch.zeros(4, 25, 25).cuda()

						# train batch = 36 / test batch = 80
						# A1 = torch.zeros(104, 25, 25).cuda()

						# test batch = 72
						A1 = torch.zeros(136, 25, 25).cuda()
						A1 = A1 + A[i]
						z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))

			y = z + y if y is not None else z

		y = self.bn(y)
		y += self.down(x)
		return self.relu(y)



class TCN_GCN_unit(nn.Module):
	def __init__(self, in_channels, out_channels, A, serial, stride=1, residual=True):
		super(TCN_GCN_unit, self).__init__()
		self.gcn1 = unit_gcn(in_channels, out_channels, A, serial)
		self.tcn1 = unit_tcn(out_channels, out_channels, serial, stride=stride)
		self.relu = nn.ReLU()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = stride
		self.serial = serial
		if not residual:
			self.residual = lambda x: 0

		elif (in_channels == out_channels) and (stride == 1):
			self.residual = lambda x: x

		else:
			self.residual = unit_tcn(in_channels, out_channels, serial, kernel_size=1, stride=stride)

	def forward(self, x):
		x = self.tcn1(self.gcn1(x)) + self.residual(x)
		return self.relu(x)
	


class Model(nn.Module):
	def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
		super(Model, self).__init__()

		if graph is None:
			raise ValueError()
		else:
			Graph = import_class(graph)
			self.graph = Graph(**graph_args)

		# A is a matrix like adjancent matrix
		
		A = self.graph.A
		
		self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

		self.l1 = TCN_GCN_unit(3, 64, A, 0, residual=False)
		self.l2 = TCN_GCN_unit(64, 64, A, 1)
		self.l3 = TCN_GCN_unit(64, 64, A, 2)
		self.l4 = TCN_GCN_unit(64, 64, A, 3)
		self.l5 = TCN_GCN_unit(64, 128, A, 4, stride=2)
		self.l6 = TCN_GCN_unit(128, 128, A, 5)
		self.l7 = TCN_GCN_unit(128, 128, A, 6)
		self.l8 = TCN_GCN_unit(128, 256, A, 7, stride=2)
		self.l9 = TCN_GCN_unit(256, 256, A, 8)
		self.l10 = TCN_GCN_unit(256, 256, A, 9)
		self.fc = nn.Linear(256, num_class)

		nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
		bn_init(self.data_bn, 1)

		self.total = 0
		self.none_zero = 0
		self.forward_cnt = 0
		self.mean_val = torch.zeros(256).cuda()
		self.layerout_total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.layerout_nozero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


	def channel_abs_distribute(self, fea, serial):
		filename = "abs_" + str(serial) + ".txt"
		a, b, c, d = fea.shape
		fea = torch.abs(fea)
		fea = fea.view(b, a*c*d)
		fea = torch.mean(fea, 1)
		fea = fea.cpu().numpy()
		f = open(filename, "w")
		for i in range (fea.shape[0]):
			f.write(str(fea[i].item()))
			f.write('\n')
		f.close()



	def forward(self, x):
		self.forward_cnt += 1

		N, C, T, V, M = x.size()

		x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
		x = self.data_bn(x)
		x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

		x = quantize(x, 5, 10)
		x = self.l1(x)
		x = quantize(x, 5, 10)
		x = self.l2(x)
		x = quantize(x, 5, 10)
		x = self.l3(x)
		x = quantize(x, 5, 10)
		x = self.l4(x)
		x = quantize(x, 5, 10)
		x = self.l5(x)
		x = quantize(x, 5, 10)
		x = self.l6(x)
		x = quantize(x, 5, 10)
		x = self.l7(x)
		x = quantize(x, 5, 10)
		x = self.l8(x)
		x = quantize(x, 5, 10)
		x = self.l9(x)
		x = quantize(x, 5, 10)
		x = self.l10(x)
		x = quantize(x, 5, 10)
		

		# x = self.l1(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 1)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[0] += total
		# self.layerout_nozero[0] += nozero

		# x = self.l2(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 2)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[1] += total
		# self.layerout_nozero[1] += nozero

		# x = self.l3(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 3)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[2] += total
		# self.layerout_nozero[2] += nozero

		# x = self.l4(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 4)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[3] += total
		# self.layerout_nozero[3] += nozero

		# x = self.l5(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 5)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[4] += total
		# self.layerout_nozero[4] += nozero

		# x = self.l6(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 6)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[5] += total
		# self.layerout_nozero[5] += nozero

		# x = self.l7(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 7)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[6] += total
		# self.layerout_nozero[6] += nozero

		# x = self.l8(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 8)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[7] += total
		# self.layerout_nozero[7] += nozero

		# x = self.l9(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 9)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[8] += total
		# self.layerout_nozero[8] += nozero

		# x = self.l10(x)
		# a = torch.clone(x).detach()
		# self.channel_abs_distribute(a, 10)
		# total, nozero = count_sparsity(a)
		# self.layerout_total[9] += total
		# self.layerout_nozero[9] += nozero


		c_new = x.size(1)
		x = x.view(N, M, c_new, -1)
		x = x.mean(3).mean(1)
		if(fc_prune_switch):
			with torch.no_grad():
				a, b = self.fc.weight.shape
				mask = torch.ones(a, b).cuda()
				mask[:, load_fc_drop_list()] = 0
			torch_prune.custom_from_mask(self.fc, name = 'weight', mask = mask)
			torch_prune.remove(self.fc, name = "weight")
			self.fc.weight.data = quantize(self.fc.weight.data, 5, 10)
			x = self.fc(x)
		else:
			self.fc.weight.data = quantize(self.fc.weight.data, 5, 10)
			x = self.fc(x)

		# if(self.forward_cnt == 262):
		# 	f = open("dcp0cov9xdcp0_feasparisty.txt", "w")
		# 	for i in range (10):
		# 		f.write(str((self.layerout_total[i] - self.layerout_nozero[i]) / self.layerout_total[i]))
		# 		f.write('\n')
		# 	f.close()


		return x
