import sys, os
sys.path.append('src')

import torch
model = torch.load('/home/cse/AC2/src/best_model.pth')
global_user_info = {}

############################################################

from pyramid.view import (view_config, view_defaults)

@view_defaults(renderer='json')
class Views:

	def __init__(self, request):
		self.request = request

	@view_config(route_name='index')
	def home(self):
		from pyramid.httpexceptions import HTTPFound
		return HTTPFound(location='public/index.html') # redirection
		# from pyramid.response import Response
		# return Response('<body>Visit <a href="/howdy">hello</a></body>')

	# @view_config(route_name='file')
	# def file(self):
	# 	file = self.request.matchdict['file']
	# 	return HTTPFound(location='public/' + file) # redirection

	@view_config(route_name='json')
	def json_data(self):
		service_name = self.request.matchdict['service_name']
		if service_name == 'initboard': return self.initboard()
		if service_name == 'isvalid': return self.isvalid()
		if service_name == 'review': return self.review()
		if service_name == 'oppomove': return self.oppomove()
		return {'error': 'No such service: %s' % service_name}

	def _user_name(self):
		import random, string
		rand_name = None
		session = self.request.session
		if 'rand_name' in session:
			rand_name = session['rand_name']
		else:
			rand_name = ''.join([random.choice(string.ascii_uppercase + string.digits) for n in range(64)])
			session['rand_name'] = rand_name
		return rand_name

	def initboard(self):
		from src.ccboard import ChessBoard
		user_name = self._user_name()
		board = ChessBoard()
		global global_user_info
		global_user_info[user_name] = { 'board' : board }
		return board.state_dict()

	def _get_board(self):
		user_name = self._user_name()
		global global_user_info
		if user_name not in global_user_info:
			return {'reply':'无玩家信息'}, None, None
		user_info = global_user_info[user_name]
		if 'board' not in user_info: return {'reply':'无正在进行的棋局'}, None, None
		board = user_info['board']
		return None, user_info, board

	def isvalid(self):
		error, user_info, board = self._get_board()
		if error is not None: return error
		params = self.request.params
		chess_pos = (int(params['pos_x']), int(params['pos_y']))
		action = (str(params['chess_type']), (int(params['mov_x']), int(params['mov_y'])))
		valid, chess_removed = board.take_action(chess_pos, action)
		if not valid: return {'reply':False}
		if board.is_terminated():
			del user_info['board']
		return {'reply':True, 'terminated':board.is_terminated(), 'winner':board.is_winner() }

	def _oppo_action(self, chess_pos, action):
		chess_pos = (8 - chess_pos[0], 9 - chess_pos[1])
		action = (action[0], (-action[1][0], -action[1][1]))
		return chess_pos, action

	def oppomove(self):
		error, user_info, board = self._get_board()
		if error is not None: return error
		board.switch_players()
		# chess_pos, action = next_action_for_evaluation(model, board)
		from pure_mcts import next_action_for_evaluation
		chess_pos, action = next_action_for_evaluation(model, board)
		
		if isinstance(action, int):
			act_list = board.action_list()
			action = act_list[action]
		board.take_action(chess_pos, action)
		board.switch_players()
		chess_pos, action = self._oppo_action(chess_pos, action)
		if board.is_terminated():
			del user_info['board']
		reply = {'pos_x':chess_pos[0], 'pos_y':chess_pos[1],
			'chess_type':action[0], 'mov_x':action[1][0], 'mov_y':action[1][1], 
			'terminated':board.is_terminated(), 'winner':board.is_winner() }
		# print(reply)
		return reply

	def _list_reviewable_games(self):
		import os
		reviewable_games = {}
		log_path = 'src/log'
		for root, dirs, files in os.walk(log_path):
			for file in files:
				if file.endswith('.actions.json'):
					game_name = file.split('.')[0]
					actions_path = os.path.join(root, file)
					reviewable_games[game_name] = actions_path
		return reviewable_games

	def _reviewable_games(self, reviewable_games):
		import json, src.misc as misc
		game_names = list(reviewable_games.keys())
		game_names.sort(reverse=True)
		for game_name in game_names:
			game_path = reviewable_games[game_name]
			actions = json.loads( misc.load_file(game_path) )
			for i in range(len(actions)):
				action = actions[i]
				if i % 2 == 1:
					action[0], action[1] = self._oppo_action(action[0], action[1])
			reviewable_games[game_name] = actions
		return game_names

	def review(self):
		from src.ccboard import ChessBoard
		reviewable_games = self._list_reviewable_games()
		game_names = self._reviewable_games(reviewable_games)
		return { 'game_names': game_names,
			'game_actions': reviewable_games,
			'init_board': ChessBoard().state_dict() }

############################################################

if __name__ == '__main__':
	from pyramid.session import SignedCookieSessionFactory
	from pyramid.config import Configurator
	from waitress import serve

	session_factory = SignedCookieSessionFactory('sysu_cl_ac_secret')

	config = Configurator(session_factory=session_factory)
	config.add_route('index', '/')
	config.add_route('json', '/json/{service_name}')
	# config.add_route('file', '/{file}')
	config.add_static_view(name='public', path='web')
	config.scan('.')
	app = config.make_wsgi_app()
	serve(app, host='0.0.0.0', port=6543)

