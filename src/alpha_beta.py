score = {}
score['king'] = 500
score['guard'] = 20
score['knight'] = 100
score['cannon'] = 200 
score['bishop'] = 50
score['pawn'] = 10
score['rock'] = 300
INF = 10000000
def evaluate_board_naive(board):
    value = 0
    for pos in board.board:
            x, y = pos
            chess = board.board[pos]
            if chess.player == 'Red':
                value = value + score[chess.chess_type]
            else:
                value = value - score[chess.chess_type]
    return value

def get_next_moves(board):
    actions, _ = board.nn_valid_actions()
    return actions
search_count = 0
def negamax(board, depth, alpha, beta, player = None):
    global search_count
    search_count = search_count + 1
    if depth == 0 or board.is_terminated():
        return evaluate_board_naive(board), None
    available_moves = get_next_moves(board)
    actions = board.action_list()
    best_move = None
    best_score = -10000000
    for chess_pos, action_index in available_moves:
        new_board = board.clone()
        action = (chess_pos, actions[action_index])
        new_board.take_action(*action)
        new_board.switch_players()
        score, move = negamax(new_board, depth - 1, -beta, -max(alpha, best_score))
        score = -score
        if score > best_score:
            best_score = score
            best_move = action
            if best_score >= beta:
                return best_score, best_move
    return best_score, best_move


