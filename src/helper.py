# -*- coding: utf-8 -*-
import torch

args_ = None

def parse_arguments():
	import argparse
	# reinforcement
	parser = argparse.ArgumentParser(description='AlphaChess training')
	parser.add_argument('--batch_size', default=1024, type=int, help='Batch size for training')
	parser.add_argument('--search_batch_size', default=128, type=int, help='Batch size for evaluation in search')
	parser.add_argument('--search_simulations', default=256, type=int, help='Number of simulations in search')
	parser.add_argument('--max_game_steps', default=128, type=int, help='Maximum number of steps in each game')
	parser.add_argument('--policy_noise_ratio', default=0.25, type=float, help='Ratio of noise added into the policy in exploration')
	parser.add_argument('--replay_buffer_size', default=500000, type=int, help='The size of the replay buffer.')
	# parser.add_argument('--evaluation_interval', default=10, type=int, help='Number of epochs between evaluations.')
	parser.add_argument('--evaluation_games', default=10, type=int, help='Number of games used in evaluation.')
	parser.add_argument('--c_puct', default=1., type=float, help='A constant determining the level of exploration.')
	# model
	parser.add_argument('--in_channels', default=15, type=int, help='Types of chess (*2 players) (+1 empty) per position.')
	parser.add_argument('--out_channels', default=96, type=int, help='Number of actions per position')
	parser.add_argument('--hidden_channels', default=256, type=int, help='Number of hidden channels')
	parser.add_argument('--residual_blocks', default=19, type=int, help='Number of residual blocks')
	parser.add_argument('--board_width', default=9, type=int, help='Width of the game board')
	parser.add_argument('--board_height', default=10, type=int, help='Height of the game board')
	# train
	parser.add_argument('--model_path', default='best.pth', help='Location to save best validation model')
	parser.add_argument('--train_gpu', default=0, type=int, help='The single device used in training.')
	parser.add_argument('--exploration_gpus', default='1,2', type=str, help='The GPU ids for the GPUs used for exploration.')
	parser.add_argument('--exploration_processes', default=5, type=int, help='The number of exploration threads per gpu.')
	parser.add_argument('--epochs', default=1000000, type=int, help='Number of training epochs')
	parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
	parser.add_argument('--anneal_interval', default=100, type=int, help='Epochs between annealing is applied')
	parser.add_argument('--learning_anneal', default=1.01, type=float, help='Annealing applied to learning rate')
	parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
	parser.add_argument('--train_mini_batch', default=128, type=int, help='Mini-batch size in training')
	parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
	parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
	parser.add_argument('--logdir', default='log', type=str, help='Log folder')
	parser.add_argument('--continue_from', type=str, help='Continue from last training epoch')
	parser.add_argument('--plot', action='store_true', help='Plot training progress in terms of accuracies')
	# parse
	global args_
	args_ = parser.parse_args()
	args_.exploration_gpus = args_.exploration_gpus.split(",")
	for i in range(len(args_.exploration_gpus)):
		args_.exploration_gpus[i] = int(args_.exploration_gpus[i])

parse_arguments()

def args():
	return args_

def _get_input(indexes):
	from ccboard import _chess_id_map
	indexes = indexes.unsqueeze(1)
	size = list(indexes.size())
	size[1] = 1 + len(_chess_id_map)
	x = indexes.new().float().resize_(size).fill_(0)
	x.scatter_(1, indexes, x.new().resize_(indexes.size()).fill_(1))
	return x

def model_predict_train(model, x, has_cuda, gpu_id):
	from torch.autograd import Variable
	model.train()
	if has_cuda:
		x = x.cuda(gpu_id)
	x = _get_input(x)
	x = Variable(x, requires_grad=False)
	p, v = model(x)
	return p, v

def model_predict_eval(model, x, valid_actions, has_cuda, gpu_id):
	from torch.autograd import Variable
	import torch.nn.functional as F
	batch_size = len(valid_actions)
	model.eval()
	if has_cuda:
		x = x.cuda(gpu_id)
	valid_actions2 = []
	for i in range(batch_size):
		valid_actions2.append(valid_actions[i])
		if has_cuda:
			valid_actions2[i] = valid_actions2[i].cuda(gpu_id)
	x = _get_input(x)
	x = Variable(x, volatile=True)
	p, v = model(x)
	p = F.softmax(p.view(batch_size, -1), dim=1)
	p, v = p.data, v.data.view(-1)
	p_list = []
	for i in range(batch_size):
		if valid_actions2[i].dim() == 0:
			p1 = torch.Tensor()
		else:
			p1 = p[i].index_select(0, valid_actions2[i])
			if has_cuda: p1 = p1.cpu()
		p_list.append(p1)
	v_list = []
	if has_cuda: v = v.cpu()
	for i in range(batch_size):
		v_list.append(v[i])
	return p_list, v_list

def get_next_moves(board):
    actions, _ = board.nn_valid_actions()
    return actions, board.action_list()