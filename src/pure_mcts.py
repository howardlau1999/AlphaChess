import numpy as np
import random
import torch
def tau_function(t):
    return 1 if t < 30 else 0

class Edge(object):
    def __init__(self, prev_node, p, a, **kwargs):
        self.prev_node = prev_node
        self.next_node = kwargs.get('next_node', None)
        self.a = a # action
        self.n = 0 # visit_count
        self.w = 0 # total_reward
        self.p = p # probability
        
    def q(self):
        return 0 if self.n == 0 else self.w / self.n

    def u(self, c_puct):
        n = 1

        for edge in self.prev_node.next_edges.values():
            n += edge.n
        from math import sqrt
        return c_puct * self.p * sqrt(n) / (1 + self.n)

class Node(object):
    def __init__(self, s, prev_edge=None, **kwargs):
        self.s = s
        self.n = kwargs.get('n', 0)
        self.v = kwargs.get('v', None)
        self.prev_edge = prev_edge
        self.next_edges = None

    def propagate(self, v):
        if self.prev_edge is None:
            return
        v = -v
        self.prev_edge.w += v
        self.prev_edge.n += 1
        self.prev_edge.prev_node.n += 1
        self.prev_edge.prev_node.propagate(v)

    def expand(self, probs, actions, v):
        self.next_edges = {}
        self.v = v
        if self.s.is_terminated():
            if not self.s.is_winner():
                self.v = -1
        else:
            for p, a in zip(probs, actions):
                self.next_edges[a] = Edge(self, p, a)
        self.propagate(self.v)

class MCTS(object):
    def __init__(self, root, model, **kwargs):
        self.root = root
        self.c_puct = kwargs.get('c_puct', 1.)
        self.model = model

    # Select the max Q+U edge
    def select(self, node):
        # We have reached a leaf node with
        # no outgoing edges
        if node.next_edges is None:
            return None
        if node.s.is_terminated():
            return None

        edges = list(node.next_edges.values())
        edges.sort(key=lambda e: e.q() + e.u(self.c_puct), reverse=True)
        
        # We choose from the max Q+U value edges
        # to see if it can reach a new status of game
        for e in edges:
            """
            We need to expand this edge
            because it is one of a node's out going
            children but it doesn't have next nodes
            """
            
            if e.next_node is None:
                s = node.s.get_next_board(*e.a)[0]
                s.switch_players()
                e.next_node = Node(s, e)
                return e.next_node
            else:
                to_expand = self.select(e.next_node)
                if to_expand is not None:
                    return to_expand
        return None

    def expand(self, batch):

        # Before introducing the neural network 
        # We use mean probability and no score
        # until we reach the end of a game


        """ Insert Neural Network here
        p = [1.0 / len(valid_actions)] * len(valid_actions)

        light_play = node.s.clone()
        actions = board.action_list()
        while not light_play.is_terminated():
            light_play.switch_players()
            available_moves, _ = get_next_moves(light_play)
            move = random.choice(available_moves)
            chess_pos, index = move
            action = (chess_pos, actions[index])
            light_play.take_action(*action)

        v = 1 if light_play.wins and light_play.get_player() == "Red" else 0
        """
        batch_size = len(batch)
        x = torch.stack([node.s.nn_board_repr() for node in batch])
        valid_actions = [node.s.nn_valid_actions() for node in batch]
        valid_actions, valid_action_indexes = zip(*valid_actions)
        from new_train import predict_eval
        p, v = predict_eval(self.model, x, valid_action_indexes)
        for i in range(batch_size):
            node = batch[i]
            node.expand(p[i], valid_actions[i], v[i])
        return node

    def simulation(self, simulations=256, batch_size=128, proc_id=None):
        queue = []
        from tqdm import tqdm
        pbar_desc = "MCTS Simulation "
        if proc_id is not None:
            pbar_desc += str(proc_id)
        pbar = tqdm(total=simulations, desc=pbar_desc)
        previous_n = 0
        if self.root.next_edges is None:
            queue.append(self.root)
        while self.root.n < simulations:
            to_expand = self.select(self.root)
            if to_expand is not None:
                queue.append(to_expand)
            if len(queue) == batch_size or to_expand is None:
                if len(queue) == 0 : break
                self.expand(queue)
                queue = []
            pbar.update(self.root.n - previous_n)
            previous_n = self.root.n
        pbar.close()
        return self._policy()

    def _policy(self):
        policy = []
        actions = []
        for action, edge in self.root.next_edges.items():
            (x, y), a = action
            n = edge.n
            # print('>', action, edge.n, edge.next_node is not None)
            if n == 0: continue
            if edge.next_node is not None and edge.next_node.s.is_terminated():
                n += 128 # winning edge is assigned a large virtual n
            policy.append(n)
            actions.append(action)
        return policy, actions


    def _print_tree(self, node=None, indent=''):
        if node is None : return
        if node.n == 0: return
        
        print(indent + 'is_terminated: ' + str(node.s.is_terminated()))
        print(indent + 'n: ' + str(node.n))
        print(indent + 'v: ' + str(node.v))
        if node.next_edges is None:
            print(indent + 'next_edges: None')
            return
        print(indent + 'edges: ' + str(len(node.next_edges)))
        
        if len(node.next_edges) == 0:
            print(indent + 'next_edges: {}')
            return
        act_list = node.s.action_list()
        for a in node.next_edges:
            edge = node.next_edges[a]
            self._print_tree(edge.next_node, indent + '  ')

    def print_tree(self):
        self._print_tree(self.root)

def next_action_for_evaluation(model, board):
    policy, actions = mcts_policy(Node(board), model, 0, 0)
    return next_action(policy, actions, 0)

def mcts_policy(node, model, tau, proc_id=None):
    tree = MCTS(node, model)
    policy, actions = tree.simulation(proc_id=proc_id)
    policy = torch.Tensor(policy)
    if tau == 0:
        policy = (policy == policy.max()).type_as(policy)
    else:
        policy = policy.pow(1 / tau)
    policy = policy / policy.sum()
    return policy, actions

def next_action(policy, actions, noise_ratio=0):
    if noise_ratio > 0:
        noise = torch.rand(policy.size())
        noise = noise * (noise_ratio / noise.sum())
        policy = policy * (1 - noise_ratio) + noise
    policy = policy.numpy()
    sum_ = policy.sum()
    policy = policy / sum_
    policy = policy.tolist()
    sum_ = (sum(policy) - 1)
    for i in range(len(policy)):
        if policy[i] > sum_:
            policy[i] -= sum_
            break
    # choose an action according to probability policy
    index = np.random.choice(range(len(policy)), size=1, p=policy)[0]
    return actions[index]

class Experience():

    def __init__(self, s, p, v, a):
        self.s = s
        self.p = p
        self.v = v
        self.a = a

def explore(board, models, max_game_steps, noise_ratio, tau_func=tau_function, policy_noise_ratio=0, resign=None, logger=None, proc_id=None):
    
    from tqdm import tqdm
    import pickle
    from ccboard import action_index
    history = []
    winner = None
    cur_node = Node(board)
    pbar_desc = "Exploration Progress "
    if proc_id is not None:
        pbar_desc += str(proc_id)
    for step in tqdm(range(max_game_steps), desc=pbar_desc):
        policy, actions = mcts_policy(cur_node, models[step % 2], tau_func(step), proc_id=proc_id)
        pos, action = next_action(policy, actions, noise_ratio)
        history.append(Experience(cur_node.s, policy, cur_node.v, action_index(pos, action)))
        
        cur_node = cur_node.next_edges[(pos, action)].next_node
        cur_node.prev_edge = None

        if cur_node.s.is_terminated():
            next_step_wins = cur_node.s.is_winner()
            winner = (step + 1 if next_step_wins else step) % 2
            break



    # fill scores
    min_winner_score = 1
    for step in range(len(history)):
        if winner is None:
            history[step].v = 0
        elif step % 2 == winner:
            if min_winner_score > history[step].v: 
                min_winner_score = history[step].v
            history[step].v = 1
        else:
            history[step].v = -1

    # save history
    with open("mcts_nn.his", "wb") as f:
        pickle.dump(history, f)

    return history, winner, min_winner_score

if __name__ == '__main__':
    from ccboard import ChessBoard
    # mcts.print_tree()
        
