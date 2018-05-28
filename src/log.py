# -*- coding: utf-8 -*-

import os, torch, misc

class Logger(object):

	def __init__(self, args, init_train_info={}, sub_dir=None):
		self.args = args		
		misc.ensure_dir(args.logdir)
		sub_dir = args.continue_from or sub_dir or misc.datetimestr()
		self.logdir = os.path.join(args.logdir, sub_dir)
		misc.ensure_dir(self.logdir)
		self._setup_log_file()
		self._create_train_info(args, init_train_info)
		self._game_info = {}

	def _setup_log_file(self):
		import logging
		log_file = os.path.join(self.logdir, 'log.txt')
		self._logger = logging.getLogger()
		self._logger.setLevel('INFO')
		fh = logging.FileHandler(log_file)
		ch = logging.StreamHandler()
		self._logger.addHandler(fh)
		self._logger.addHandler(ch)

	def info(self, str_info):
		self._logger.info(str_info)

	# train_info


	def _create_train_info(self, args, init_train_info):
		import json
		train_info_path = os.path.join(self.logdir, 'train_info.txt')
		if os.path.isfile(train_info_path):
			self.train_info = json.loads(misc.load_file(train_info_path))
		else:
			self.train_info = init_train_info

	def save_train_info(self):
		import json
		train_info_path = os.path.join(self.logdir, 'train_info.txt')
		misc.save_file(train_info_path, json.dumps(self.train_info))

	def plot_progress(self):
		import plot
		pdf_plot_path = os.path.join(self.logdir, 'progress.pdf')
		x_data = torch.Tensor(range(1, 1 + len(self.train_info['loss'])))
		y_data = torch.Tensor([self.train_info['loss'],])
		# y_data, y_err = plot.smooth2d(y_data, step=3)
		plot.plot(x_data, y_data, y_err=None, legends=None, 
				title=None, xlabel='Epoch', ylabel='Loss', filename=pdf_plot_path)

	# model

	def _model_config(self):
		config = {}
		config['in_channels'] = self.args.in_channels
		config['out_channels'] = self.args.out_channels
		config['hidden_channels'] = self.args.hidden_channels
		config['residual_blocks'] = self.args.residual_blocks
		config['board_size'] = (self.args.board_width, self.args.board_height)
		return config

	@staticmethod
	def lastest_model_path(logdir):
		_, folders, _ = next(os.walk(logdir))
		root, _, files = next(os.walk(os.path.join(logdir, folders[-1])))
		files = [file for file in files if file.endswith('.pth')]
		return os.path.join(root, files[-1])

	@staticmethod
	def _init_model(model_config, gpu_id, state_dict=None):
		from model import AlphaGoModel
		model_ = AlphaGoModel(**model_config)
		if state_dict is not None:
			model_.load_state_dict(state_dict)
		if torch.cuda.is_available():
			model_ = model_.cuda(gpu_id)
		return model_

	@staticmethod
	def _load_model(path, gpu_id):
		print('Loading model from: %s' % path)
		package = torch.load(path, map_location=lambda storage, location: storage)
		return Logger._init_model(package['config'], gpu_id, package['state_dict'])

	def get_model_path(self):
		if 'model_path' not in self.train_info: return None
		return os.path.join(self.logdir, self.train_info['model_path'])

	def create_model(self, gpu_id):
		import model
		if 'model_path' in self.train_info:
			model_path = os.path.join(self.logdir, self.train_info['model_path'])
			if os.path.exists(model_path):
				return Logger._load_model(model_path, gpu_id)
		return Logger._init_model(self._model_config(), gpu_id)

	def save_model(self, model, model_path=None):
		to_remove = None
		if model_path is None:
			# remove old
			if 'model_path' in self.train_info:
				model_path = os.path.join(self.logdir, self.train_info['model_path'])
				if os.path.exists(model_path):
					to_remove = model_path
					print("Removing old model {}".format(self.train_info['model_path']))
			# new path name
			model_path = misc.datetimestr() + '.model.pth'
			self.train_info['model_path'] = model_path
		print("Saving model to {}".format(model_path))
		package = {
			'config': self._model_config(),
			'state_dict': model.state_dict(),
		}
		torch.save(package, os.path.join(self.logdir, model_path))
		if to_remove is not None:
			os.remove(to_remove)


	def copy_model(self, from_model, to_model):
		to_model.load_state_dict(from_model.state_dict())

	def clone_model(self, from_model, gpu_id):
		to_model = Logger._init_model(self._model_config(), gpu_id)
		self.copy_model(from_model, to_model)
		return to_model

	# game

	def log_game_action(self, game_name, pos, action):
		if game_name not in self._game_info:
			self._game_info[game_name] = {}
		game = self._game_info.get(game_name)
		if 'actions' not in game:
			game['actions'] = []
		game['actions'].append((pos, action))

	def end_log_game_action(self, game_name):
		import json
		if game_name not in self._game_info: return
		game = self._game_info.get(game_name)
		if 'actions' in game:
			file_path = os.path.join(self.logdir, 'game_info')
			misc.ensure_dir(file_path)
			file_path = os.path.join(file_path, game_name + '.actions.json')
			misc.save_file(file_path, json.dumps(game['actions']))
		del self._game_info[game_name]
