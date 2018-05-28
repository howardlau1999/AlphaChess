
function playable_chess_board(board_id, setting) {
	var board = {}
	board.board_id = board_id
	board.board_map = {}
	_draw_plate(board, setting)

	send_request('/json/initboard', {}, function(board_map, textStatus) {
		board.board_map = board_map
		_draw_chesses(board, true)
	});
}

function _draw_plate(board, setting) {
	if (setting === 'large') {
		setting = {};
		setting.offset_x = 29;
		setting.offset_y = 29-3;
		setting.grid_size_x = 59;
		setting.grid_size_y = 61;
		setting.plate_size = 535;
		setting.chess_size = 55;
	}
	else if (setting === 'middle') {
		setting = {};
		setting.offset_x = 17;
		setting.offset_y = 17-3;
		setting.grid_size_x = 33;
		setting.grid_size_y = 34;
		setting.plate_size = 300;
		setting.chess_size = 30;
	}
	else if (setting === 'small') {
		setting = {};
		setting.offset_x = 11;
		setting.offset_y = 12-4;
		setting.grid_size_x = 22;
		setting.grid_size_y = 22.6;
		setting.plate_size = 200;
		setting.chess_size = 20;
	}
	else if (setting === 'tiny') {
		setting = {};
		setting.offset_x = 5.5;
		setting.offset_y = 5-3;
		setting.grid_size_x = 11;
		setting.grid_size_y = 11.4;
		setting.plate_size = 100;
		setting.chess_size = 11;
	}
	board.setting = setting

	var image = document.createElement("img");
	image.src = "img/plate.png";
	// image.style.position = "absolute";
	// image.style.left = 0;
	// image.style.top = 0;
	image.draggable = false;
	image.style.opacity = 0.8;
	image.style.width = board.setting.plate_size;
	var parent = document.getElementById(board.board_id);
	parent.appendChild(image);
	// parent.style = "left:0;top:0;height:" + image.height + ";width:" + image.width + ";position:relative;" + parent.style
	parent.style = "left:0;top:0;position:relative;vertical-align:top;" + parent.style
}

function _draw_chesses(board, playable) {
	_clear_chess_board(board.board_id, "Red");
	_clear_chess_board(board.board_id, "Black");
	var parent = document.getElementById(board.board_id);
	for (pos in board.board_map) {
		chess = board.board_map[pos];
		image = get_chess_image(board.setting, chess[0], chess[1], chess[2], chess[3]);
		chess[4] = image;
		parent.appendChild(image);
	}
	freeze_all_chesses(board, "Black", true);
	freeze_all_chesses(board, "Red", ! playable);
}

function get_chess_image(setting, x, y, chess_player, chess_type) {
	pos = chess_index_to_position(setting, x, y)
	var image = document.createElement("img");
	// image.setAttribute("src", "img/" + chess_player + "_" + chess_type + ".png");
	image.src = "img/" + chess_player + "_" + chess_type + ".png";
	// image.setAttribute("class", "chessman");
	image.className = chess_player;
	// image.setAttribute("style", "position:absolute;left:" + pos[0] + "px;top:" + pos[1] + "px;width:55px;");	
	image.style.position = "absolute";
	image.style.left = pos[0];
	image.style.top = pos[1];
	image.style.width = setting.chess_size;
	image.style.opacity = 0.95;
	image.draggable = true;
	return image;
}

function chess_index_to_position(setting, x, y) {
	offset_x = setting.offset_x - setting.chess_size / 2;
	offset_y = setting.offset_y - setting.chess_size / 2;
	var left = offset_x + x * setting.grid_size_x;
	var top = offset_y + (9 - y) * setting.grid_size_y;
	return [left, top];
}

function get_distance(x1, y1, x2, y2) {
	var dx = x1 - x2;
	var dy = y1 - y2;
	return Math.sqrt(dx * dx + dy * dy);
}

function chess_position_to_index(setting, x, y) {
	var index = null;
	var min_dis = null;
	for (var i = 0; i < 9; ++ i) {
		for (var j = 0; j < 10; ++ j) {
			pos = chess_index_to_position(setting, i, j)
			dis = get_distance(x, y, pos[0], pos[1])
			if (index == null || dis < min_dis) {
				index = [i, j];
				min_dis = dis;
			}
		}
	}
	return index;
}

function _clear_chess_board(board_id, className="Red") {
	let parent = document.getElementById(board_id);
	let childNodes = parent.getElementsByClassName(className);
	for (var i = childNodes.length - 1; i >= 0; -- i) {
		childNodes[i].parentNode.removeChild(childNodes[i]);
	}
}

function handleDragStart(board, e) {
	board.drag_start_x = e.x;
	board.drag_start_y = e.y;
	board.drag_start_elem_x = e.srcElement.offsetLeft;
	board.drag_start_elem_y = e.srcElement.offsetTop;
}

function get_action(board, old_index, new_index) {
	var move = [new_index[0] - old_index[0], new_index[1] - old_index[1]];
	if (move[0] == 0 && move[1] == 0) {
		return null;
	}
	index = old_index[0] + "_" + old_index[1];
	chess = board.board_map[index];
	return { 'pos_x':old_index[0], 'pos_y':old_index[1], 'mov_x':move[0], 'mov_y':move[1], 'chess_type':chess[3] };
}

function _move_chess(board, old_index, new_index) {
	var key1 = old_index[0] + "_" + old_index[1];
	var key2 = new_index[0] + "_" + new_index[1];
	// show_message("Keys", key1 + ' -> ' + key2);
	var chess = board.board_map[key1];
	var pos = chess_index_to_position(board.setting, new_index[0], new_index[1])
	chess_image = chess[4];
	chess_image.id = "animating";
	$("#animating").animate({
		left:pos[0]+'px',
		top:pos[1]+'px',
	}, 100);
	chess_image.id = "";
	// chess_image.style.left = pos[0];
	// chess_image.style.top = pos[1];
	delete board.board_map[key1];
	chess[0] = new_index[0];
	chess[1] = new_index[1];
	old_chess = board.board_map[key2];
	board.board_map[key2] = chess;
	if (old_chess != null) {
		old_chess[4].parentNode.removeChild(old_chess[4]);
	}
	return old_chess;
}

function handleDragEnd(board, e) {
	new_left = board.drag_start_elem_x + (e.x - board.drag_start_x);
	new_top = board.drag_start_elem_y + (e.y - board.drag_start_y);
	old_index = chess_position_to_index(board.setting, board.drag_start_elem_x, board.drag_start_elem_y);
	new_index = chess_position_to_index(board.setting, new_left, new_top);
	action = get_action(board, old_index, new_index);
	if (action == null) return;
	send_request("/json/isvalid", action, function(data, textStatus) {
		if (data['reply'] == true) {
			_move_chess(board, old_index, new_index);
			if (data['terminated'] == true) {
				msg_title = (data['winner'] == true ? '你赢了 :)' : '你输了 :(')
				show_message(msg_title, '<p>游戏已结束，请刷新网页重新开始一局。</p>');
			}
			else {
				send_request("/json/oppomove", {}, function(data, textStatus) {
					old_index = [ data['pos_x'], data['pos_y'] ];
					new_index = [ old_index[0] + data['mov_x'], old_index[1] + data['mov_y'] ];
					// show_message('_move_chess', old_index + ' -> ' + new_index);
					_move_chess(board, old_index, new_index);
					if (data['terminated'] == true) {
						msg_title = (data['winner'] == true ? '你赢了 :)' : '你输了 :(')
						show_message(msg_title, '<p>游戏已结束，请刷新网页重新开始一局。</p>');
					}
				});
			}
		}
		else if (data['reply'] != false) {
			show_message('错误', data['reply']);
		}
	});
}

function show_message(title, html_content) {
	$("#dialog-title").text(title);
	$("#dialog-content").html(html_content);
	$("#dialog-box").modal();
}

function send_request(request_url, reqeust_data, success_callback, fadeStart=500) {
	var faded = false;
	var fadeDuration = 100
	$.ajax({
		type: "POST",
		url: request_url,
		data: reqeust_data,
		dataType: 'json',
		beforeSend: function(XMLHttpRequest) {
			if (fadeStart > 0) {
				setTimeout(function() {
					if (fadeStart > 0) {
						$('.block_bg').fadeIn(fadeDuration);
						faded = true;
					}
				}, fadeStart);
			}
		},
		success: function(data, textStatus) {
			if (faded) {
				setTimeout(function() {
					success_callback(data, textStatus);
				}, fadeDuration);
			}
			else {
				success_callback(data, textStatus);
			}
		},
		complete: function(XMLHttpRequest, textStatus) {
			fadeStart = 0;
			if (faded) {
				$('.block_bg').fadeOut(fadeDuration);
			}
		},
		error: function() {
			show_message('错误', '<p>不能链接服务器.</p>');
		},
	});
}

function freeze_all_chesses(board, className="Red", freeze=true) {
	let parent = document.getElementById(board.board_id);
	let childNodes = parent.getElementsByClassName(className);
	for (var i = childNodes.length - 1; i >= 0; -- i) {
		// console.log(pos, chess)
		if (freeze) {
			childNodes[i].draggable = false;
			childNodes[i].removeEventListener('dragstart', function(e) { handleDragStart(board, e) });
			childNodes[i].removeEventListener('dragend', function(e) { handleDragEnd(board, e) });
		}
		else {
			childNodes[i].draggable = true;
			// childNodes[i].style.cursor = 'move';
			childNodes[i].addEventListener('dragstart', function(e) { handleDragStart(board, e) }, false);
			childNodes[i].addEventListener('dragend', function(e) { handleDragEnd(board, e) }, false);
		}
	}
}

///////////////////////////////////////
// review

function clear_all_children(elem_id) {
	var node = document.getElementById(elem_id);
	// node.innerHTML = "";
	// show_message('ERROR', 'No such element: ' + node + elem_id)
	while (node.hasChildNodes()) {
		node.removeChild(node.lastChild);
	}
}

function review_chess_board(board_id, setting, board_map) {
	var board = {}
	board.board_id = board_id
	board.board_map = board_map
	_draw_plate(board, setting)
	_draw_chesses(board, false)
	return board
}

function clone_object(obj) {
	return JSON.parse(JSON.stringify(obj))
}

function next_move(delay, board, actions, index) {
	if (index == actions.length) return;

	var pos = actions[index][0];
	var chess_type = actions[index][1][0];
	var action = actions[index][1][1];
	var old_index = pos;
	var new_index = [old_index[0] + action[0], old_index[1] + action[1]];
	var chess = board.board_map[pos[0] + '_' + pos[1]];
	var chess_image = chess[4]
	if (! document.contains(chess_image)) return;
	_move_chess(board, old_index, new_index);
	setTimeout(function() { next_move(delay, board, actions, index + 1); }, delay);
}

function show_actions(board_id, setting, init_board, actions, delay) {
	clear_all_children(board_id);
	var board = review_chess_board(board_id, setting, clone_object(init_board));
	var index = 0;
	setTimeout(function() { next_move(delay, board, actions, 0); }, delay);
}

function create_button(game_name, actions, board_id, setting, init_board, delay) {
	var button = document.createElement("a");
	button.text = game_name;
	button.className = "dropdown-item";
	button.onclick = function() {
		console.log(game_name)
		// console.log(actions)
		show_actions(board_id, setting, init_board, actions, delay);
	}
	return button;
}

function review(list_id, board_id, setting, delay) {

	function handler(data, textStatus) {
		if ('game_names' in data) {
			game_names = data['game_names'];
			game_actions = data['game_actions'];
			init_board = data['init_board'];

			clear_all_children(list_id);
			clear_all_children(board_id);
			var parent = document.getElementById(list_id);
			for (var i = 0; i < game_names.length; ++ i) {
				var game_name = game_names[i];
				var actions = game_actions[game_name];
				parent.appendChild(create_button(game_name, actions, board_id, setting, init_board, delay));
			}
			if (game_names.length > 0) {
				var game_name = game_names[0];
				show_actions(board_id, setting, init_board, game_actions[game_name], delay);
			}
		}
		else {
			show_message('错误', '<pre>' + iterate_object(data, '') + '</pre>');
		}
	}

	send_request("/json/review", {}, handler);

}


function reviewall(board_area_id, setting, delay) {

	function handler2(data, textStatus) {
		if ('game_names' in data) {
			game_names = data['game_names'];
			game_actions = data['game_actions'];
			init_board = data['init_board'];

			var parent = document.getElementById(board_area_id);
			for (var i = 0; i < game_names.length; ++ i) {
				var board_elem = document.createElement("span");
				parent.appendChild(board_elem);
				game_name = game_names[i];
				board_elem.id = game_name;
				show_actions(game_name, setting, clone_object(init_board), game_actions[game_name], delay);
			}
		}
		else {
			show_message('错误', '<pre>' + iterate_object(data, '') + '</pre>');
		}
	}

	send_request("/json/review", {}, handler2);
}

