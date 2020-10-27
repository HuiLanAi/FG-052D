import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import os

sample_num = 0
x_len_l10 = 0
x_len_l1 = 0
x_len_l0 = 0
x_len_l2 = 0
x_len_l3 = 0
x_len_l4 = 0
x_len_l5 = 0
x_len_l6 = 0
x_len_l7 = 0
x_len_l8 = 0
x_len_l9 = 0
x_zero_cnt_l10 = 0
x_zero_cnt_l0 = 0
x_zero_cnt_l1 = 0
x_zero_cnt_l2 = 0
x_zero_cnt_l3 = 0
x_zero_cnt_l4 = 0
x_zero_cnt_l5 = 0
x_zero_cnt_l6 = 0
x_zero_cnt_l7 = 0
x_zero_cnt_l8 = 0
x_zero_cnt_l9 = 0
x_zero_cnt_l10 = 0
x_zero_cnt_l1 = 0
x_zero_cnt_l2 = 0
x_zero_cnt_l3 = 0
x_zero_cnt_l4 = 0
x_zero_cnt_l5 = 0
x_zero_cnt_l6 = 0
x_zero_cnt_l7 = 0
x_zero_cnt_l8 = 0
x_zero_cnt_l9 = 0



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


class unit_tcn(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
		super(unit_tcn, self).__init__()
		pad = int((kernel_size - 1) / 2)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
							  stride=(stride, 1))

		self.bn = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		conv_init(self.conv)
		bn_init(self.bn, 1)

	def forward(self, x):
		x = self.bn(self.conv(x))
		return x


class unit_gcn(nn.Module):
	def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
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
		
		# A = A + B
		# A.size() = 3, 25, 25
		A = self.A.cuda(x.get_device())
		A = A + self.PA

		y = None

		for i in range(self.num_subset):
			# print(A1.size())
			# conv_a[0](x).size() = 2, 16, 300, 25
			# 2, 25, 4800
			A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
			# print(A2.size())
			# 2, 4800, 25
			A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)

			# 计算C
			# matmul(A1, A2).size() = 2, 25, 25
			# A1.size() = 2, 25, 25
			# 对倒数第2维进行softmax，归一到0~1的范围中
			A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V

			# A1 = A + B + C
			# A1.size() = 2, 25, 25
			A1 = A1 + A[i]
			
			# 2, 900, 25
			A2 = x.view(N, C * T, V)

			z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
			y = z + y if y is not None else z

		y = self.bn(y)
		y += self.down(x)
		return self.relu(y)


class TCN_GCN_unit(nn.Module):
	def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
		super(TCN_GCN_unit, self).__init__()
		self.gcn1 = unit_gcn(in_channels, out_channels, A)
		self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
		self.relu = nn.ReLU()
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

		self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
		self.l2 = TCN_GCN_unit(64, 64, A)
		self.l3 = TCN_GCN_unit(64, 64, A)
		self.l4 = TCN_GCN_unit(64, 64, A)
		self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
		self.l6 = TCN_GCN_unit(128, 128, A)
		self.l7 = TCN_GCN_unit(128, 128, A)
		self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
		self.l9 = TCN_GCN_unit(256, 256, A)
		self.l10 = TCN_GCN_unit(256, 256, A)
		self.fc = nn.Linear(256, num_class)


		print("cnm xhs")
		print("cnm xhs")
		print("cnm xhs")

		# for name, parameters in self.named_parameters():
		# 	print(name, ':', parameters.size())

		# self.l1 = TCN_GCN_unit(6, 128, A, residual=False)
		# self.l2 = TCN_GCN_unit(128, 128, A)
		# self.l3 = TCN_GCN_unit(128, 128, A)
		# self.l4 = TCN_GCN_unit(128, 128, A)
		# self.l5 = TCN_GCN_unit(128, 256, A, stride=2)
		# self.l6 = TCN_GCN_unit(256, 256, A)
		# self.l7 = TCN_GCN_unit(256, 256, A)
		# self.l8 = TCN_GCN_unit(256, 512, A, stride=2)
		# self.l9 = TCN_GCN_unit(512, 512, A)
		# self.l10 = TCN_GCN_unit(512, 512, A)

		# self.fc = nn.Linear(512, num_class)

		nn.init.normal(self.fc.weight, 0, math.sqrt(2. / num_class))
		bn_init(self.data_bn, 1)

	def forward(self, x):
		global sample_num
		global x_len_l10
		global x_len_l0
		global x_len_l1
		global x_len_l2
		global x_len_l3
		global x_len_l4
		global x_len_l5
		global x_len_l6
		global x_len_l7
		global x_len_l8
		global x_len_l9
		global x_zero_cnt_l10
		global x_zero_cnt_l0
		global x_zero_cnt_l1
		global x_zero_cnt_l2
		global x_zero_cnt_l3
		global x_zero_cnt_l4
		global x_zero_cnt_l5
		global x_zero_cnt_l6
		global x_zero_cnt_l7
		global x_zero_cnt_l8
		global x_zero_cnt_l9

		N, C, T, V, M = x.size()

		x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
		x = self.data_bn(x)
		x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l0 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l0 += 1

		x = self.l1(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l1 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l1 += 1

		x = self.l2(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l2 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l2 += 1

		x = self.l3(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l3 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l3 += 1

		x = self.l4(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l4 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l4 += 1

		x = self.l5(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l5 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l5 += 1

		x = self.l6(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l6 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l6 += 1

		x = self.l7(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l7 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l7 += 1

		x = self.l8(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l8 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l8 += 1

		x = self.l9(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l9 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l9 += 1

		x = self.l10(x)
		# x_sp = x.cpu().numpy()
		# x_sp = x_sp.flatten()
		# x_sp = x_sp.tolist()
		# x_len_l10 += len(x_sp)
		# for x_single in x_sp:
		# 	if x_single == 0:
		# 		x_zero_cnt_l10 += 1

		# if sample_num == 5000:
		# 	sparase_file = open("sparase_file.txt", "w")
		# 	sparase_file.write(str(x_zero_cnt_l0 / x_len_l0) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l1 / x_len_l1) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l2 / x_len_l2) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l3 / x_len_l3) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l4 / x_len_l4) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l5 / x_len_l5) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l6 / x_len_l6) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l7 / x_len_l7) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l8 / x_len_l8) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l9 / x_len_l9) + '\n')
		# 	sparase_file.write(str(x_zero_cnt_l10 / x_len_l10) + '\n')
		# 	sparase_file.close()
		# 	return None
		# else:
		# 	sample_num += 1
		
		# N*M,C,T,V
		c_new = x.size(1)
		x = x.view(N, M, c_new, -1)
		x = x.mean(3).mean(1)
		x = self.fc(x)

		return x
