import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import os

weight_cnt = 1
zero_weight_cnt = 0
out_flag = 0
prune_switch = False
threshold = 0.015
prune_seg_len = 4
use_ck = False


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


def prune(weight):
	global weight_cnt
	global zero_weight_cnt
	global threshold
	global out_flag
	pruned_weight = weight
	pruned_weight[torch.abs(pruned_weight) < threshold] = 0

	if(out_flag == 0):
		o_c, in_c, w, h = weight.shape
		weight_cnt += o_c * in_c * w * h
		zero_mtx = torch.zeros(o_c, in_c, w, h).cuda()
		zero_weight_cnt += torch.sum(torch.eq(pruned_weight, zero_mtx))
	
	return pruned_weight



def max_seg_prune(weight):
	o_c, in_c, w, h = weight.shape
	
	if(in_c == 3):
		return weight

	else:
		with torch.no_grad():
			pruned_weight = weight.permute(0, 2, 3, 1)
			pruned_weight = pruned_weight.reshape(o_c * w, h, in_c)
			pruned_weight = pruned_weight.contiguous().view(o_c * w * h * in_c // 4, 4)

			index_0dim = [a for a in range (0, o_c * w * h * in_c // 4)]
			index_0dim = np.array(index_0dim)
			index_0dim = torch.from_numpy(index_0dim)
			
			max_val = torch.rand((o_c * w * h * in_c // 4, 4))
			
			val, index = torch.max(torch.abs(pruned_weight), 1)
			val = val.reshape(o_c * w * h * in_c // 4, 1)
			max_val[:, 0] = val[:, 0]
			max_val[:, 1] = val[:, 0]
			max_val[:, 2] = val[:, 0]
			max_val[:, 3] = val[:, 0]

			tmp = torch.abs(pruned_weight) < max_val.cuda()
			pruned_weight[tmp] = 0

			pruned_weight = pruned_weight.contiguous().view(o_c * w * h, in_c)
			pruned_weight = pruned_weight.contiguous().view(o_c, w, h, in_c)
	
		pruned_weight_grad = pruned_weight.permute(0, 3, 1, 2)
		return pruned_weight_grad


def fc_max_seg_prune(weight):
	cat, in_c = weight.shape
	
	with torch.no_grad():
		pruned_weight = weight.contiguous().view(cat * in_c // 4, 4)

		index_0dim = [a for a in range (0, cat * in_c // 4)]
		index_0dim = np.array(index_0dim)
		index_0dim = torch.from_numpy(index_0dim)
		
		max_val = torch.rand((cat * in_c // 4, 4))
		
		val, index = torch.max(torch.abs(pruned_weight), 1)
		val = val.reshape(cat * in_c // 4, 1)
		max_val[:, 0] = val[:, 0]
		max_val[:, 1] = val[:, 0]
		max_val[:, 2] = val[:, 0]
		max_val[:, 3] = val[:, 0]

		tmp = torch.abs(pruned_weight) < max_val.cuda()
		pruned_weight[tmp] = 0

	pruned_weight_grad = pruned_weight.contiguous().view(cat, in_c)
	return pruned_weight_grad




def load_drop_list(i):
	data = []
	file_prefix = "/home/winter/2s-AGCN/model/drop/"
	file_name = file_prefix + "L" + str(i) + "_drop.txt"
	f = open(file_name, "r")
	for num in f.readlines():
		data.append(num)
	f.close()
	return data



def channel_drop_prune(weight, serial):
	if(serial == 0):
		return weight
	else:
		with torch.no_grad():
			weight[:, load_drop_list(serial), :, :] = 0
		
		pruned_weight = weight
		return pruned_weight






class unit_tcn(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
		super(unit_tcn, self).__init__()
		self.pad = int((kernel_size - 1) / 2)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(self.pad, 0),
							stride=(stride, 1))
		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		self.stride = stride
		conv_init(self.conv)
		bn_init(self.bn, 1)


	def forward(self, x):
		# if(prune_switch):
		# 	x = F.conv2d(x, max_seg_prune(self.conv.weight), self.conv.bias, padding = (self.pad, 0), stride = (self.stride, 1))
		# else:
		# 	x = self.conv(x)
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
		y = None

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
					A1 = torch.zeros(64, 25, 25).cuda()
					# A1 = torch.zeros(800, 25, 25).cuda()
					A1 = A1 + A[i]
					if(prune_switch):
						z = F.conv2d(torch.matmul(A2, A1).view(N, C, T, V), channel_drop_prune(self.conv_d[i].weight, self.serial), self.conv_d[i].bias)
					else:
						z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
				except RuntimeError:
					try:
						A1 = torch.zeros(60, 25, 25).cuda()
						A1 = A1 + A[i]
						if(prune_switch):
							z = F.conv2d(torch.matmul(A2, A1).view(N, C, T, V), channel_drop_prune(self.conv_d[i].weight, self.serial), self.conv_d[i].bias)
						else:
							z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
					except RuntimeError:
						A1 = torch.zeros(4, 25, 25).cuda()
						# A1 = torch.zeros(264, 25, 25).cuda()
						A1 = A1 + A[i]
						if(prune_switch):
							z = F.conv2d(torch.matmul(A2, A1).view(N, C, T, V), channel_drop_prune(self.conv_d[i].weight, self.serial), self.conv_d[i].bias)
						else:
							z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))

			y = z + y if y is not None else z

		y = self.bn(y)
		y += self.down(x)
		return self.relu(y)



class TCN_GCN_unit(nn.Module):
	def __init__(self, in_channels, out_channels, A, serial, stride=1, residual=True):
		super(TCN_GCN_unit, self).__init__()
		self.gcn1 = unit_gcn(in_channels, out_channels, A, serial)
		self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
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
			self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

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

	def forward(self, x):
		N, C, T, V, M = x.size()

		x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
		x = self.data_bn(x)
		x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

		x = self.l1(x)
		x = self.l2(x)
		x = self.l3(x)
		x = self.l4(x)
		x = self.l5(x)
		x = self.l6(x)
		x = self.l7(x)
		x = self.l8(x)
		x = self.l9(x)
		x = self.l10(x)

		c_new = x.size(1)
		x = x.view(N, M, c_new, -1)
		x = x.mean(3).mean(1)
		# if(prune_switch):
		# 	x = F.linear(x, fc_max_seg_prune(self.fc.weight), self.fc.bias)
		# else:
		# 	x = self.fc(x)
		x = self.fc(x)


		return x
