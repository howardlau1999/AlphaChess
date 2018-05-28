import matplotlib as mpl
mpl.use('Agg')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import torch.cuda
import numpy as np
from model import AlphaChessModel
from pure_mcts import explore
from ccboard import ChessBoard
max_steps = 128
evaluation_interval = 100
batch_size = 1024
def nn_input(indexes):
    from ccboard import _chess_id_map
    indexes = indexes.unsqueeze(1)
    size = list(indexes.size())
    size[1] = 1 + len(_chess_id_map)
    x = indexes.new().float().resize_(size).fill_(0)
    x.scatter_(1, indexes, x.new().resize_(indexes.size()).fill_(1))
    return x

def predict_train(model, x):
    model.train()
    if torch.cuda.is_available():
        x = x.cuda()
    x = nn_input(x)
    x = Variable(x, requires_grad=False)
    p, v = model(x)
    return p, v

def predict_eval(model, x, _valid_actions):
    batch_size = len(_valid_actions)
    model.eval()
    x = nn_input(x)
    if torch.cuda.is_available():
        x = x.cuda()
    valid_actions = []
    for i in range(batch_size):
        valid_actions.append(_valid_actions[i])
        if torch.cuda.is_available():
            valid_actions[i] = valid_actions[i].cuda()
    x = Variable(x, volatile=True)
    p, v = model(x)
    p = F.softmax(p.view(batch_size, -1), dim=1)
    p, v = p.data, v.data.view(-1)
    policies = []
    for i in range(batch_size):
        if valid_actions[i].dim() == 0:
            p_ = torch.Tensor()
        else:
            p_ = p[i].index_select(0, valid_actions[i])
        policies.append(p_)
    values = []
    for i in range(batch_size):
        values.append(v[i])
    return policies, values

def train_epoch(replay_buffer, optimizer, model, max_norm):
    batch_size = min(1024, len(replay_buffer))
    import random
    batch = random.sample(replay_buffer, batch_size)
    x = torch.stack([h.s.nn_board_repr() for h in batch])
    # cross_entropy requires a class index, not the full array
    target_p = torch.Tensor([h.a for h in batch]).long()
    target_v = torch.Tensor([h.v for h in batch]).unsqueeze(-1)
    return learn(x, target_p, target_v, optimizer, model, max_norm)

def learn(x, target_p, target_v, optimizer, model, max_norm=400):
    mini_batch = 128
    batch_size = x.size(0)
    optimizer.zero_grad()
    total_lost = 0
    if torch.cuda.is_available():
        target_p = target_p.cuda()
        target_v = target_v.cuda()
    from tqdm import tqdm
    for i in tqdm(range((batch_size - 1) // mini_batch + 1), desc="Mini batches"):
        start = i * mini_batch 
        end = min((i + 1) * mini_batch + 1, batch_size)
        p, v = predict_train(model, x[start:end])
        loss = model.loss(p, v, Variable(target_p[start:end]), Variable(target_v[start:end]))
        loss.backward()
        total_lost += loss.data[0]
    torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)
    optimizer.step()
    return total_lost

def train(replay_buffer, model, queue):
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=1e-4, momentum=0.9, nesterov=True)
    max_norm = 400
    loss_history = []
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    eval_iters = 1
    best_model = model
    torch.save(best_model, 'best_model.pth')
    for epoch in tqdm(range(500), desc="Training epoch"):
        if (epoch + 1) % 100 == 0:
            optim_state = optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 1.01
            optimizer.load_state_dict(optim_state)
        
        _history, _, _ = explore(ChessBoard(), [best_model, best_model], max_steps, 0.25)
        replay_buffer.extend(_history)
        # while not queue.empty():
        #    replay_buffer.extend(queue.get())
        while len(replay_buffer) > 100000: replay_buffer.pop()
        loss = train_epoch(replay_buffer, optimizer, model, max_norm)
        import misc
        print('\nTrain epoch: %d, Buffer:%d, Time: %s, Loss: %.5f' % (epoch, len(replay_buffer), misc.datetimestr(), loss))
        loss_history.append(loss)
        plt.clf()
        plt.plot(loss_history)
        plt.savefig("loss.png")

        if (epoch + 1) % evaluation_interval == 0:
            eval_iters += 1
            best_model = evaluation(eval_iters, best_model, model)
        with open('loss.his', 'wb') as f:
            import pickle
            pickle.dump(loss, f)

def compare_models(iters, eval_model, best_model, evaluation_games=10):
    wins = 0
    from tqdm import tqdm
    from pure_mcts import explore
    for i in tqdm(range(evaluation_games), desc='Comparing models'):
        _, winner, _ = explore(ChessBoard(), [best_model, eval_model], max_steps, 0.25)
        if winner is not None:
            wins += winner
    win_rate = wins / evaluation_games
    return win_rate

def evaluation(iters, best_model, model):
    eval_model = get_new_model()
    eval_model.load_state_dict(model.state_dict())
    win_rate = compare_models(iters, eval_model, best_model)
    print("\nEvaluation {} Winning rate: {}".format(iters, win_rate))
    torch.save(best_model, 'best_model.pth')
    if win_rate < 0.55: return best_model
    best_model = eval_model
    torch.save(best_model, 'best_model.pth')
    return best_model

def get_new_model():
    model = AlphaChessModel(in_channels=15, out_channels=96, hidden_channels=256, residual_blocks=19, board_size=(9, 10))
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def exploration_process_func(queue, proc_id):
    from pure_mcts import explore
    while True:
        best_model = torch.load('best_model.pth')
        history, _, _ = explore(ChessBoard(), [best_model, best_model], max_steps, 0.25, proc_id=proc_id)
        queue.put(history)

def start_exploration_processes(ctx, queue):
    for i in range(4):
        process = ctx.Process(target=exploration_process_func, args=(queue, i))
        process.daemon = True
        process.start()

def main():
    
    model = get_new_model()
    import os
    if not os.path.exists('best_model.pth'):
        torch.save(model, 'best_model.pth')

    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    # start_exploration_processes(ctx, queue)
    replay_buffer = []
    
    print("CUDA? {}".format(torch.cuda.is_available()))
    train(replay_buffer, model, queue)
    """
    try:
        
        while True:
            while not queue.empty():
                replay_buffer.extend(queue.get())
            if len(replay_buffer) >= batch_size: break
        
    finally:
        queue.close()
    """

if __name__ == '__main__':
    main()
