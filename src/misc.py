# -*- coding: utf-8 -*-


def prepare_single_device(args):
    import torch
    torch.cuda.set_device(args.gpu)


class Average(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ensure_dir(path, erase_old=False):
    import os, shutil
    if os.path.exists(path) and erase_old:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def datetimestr():
    def get_text(value):
        if value < 10:
            return '0%d' % value
        else:
            return '%d' % value
    import datetime
    now = datetime.datetime.now()
    return get_text(now.year) + get_text(now.month) + get_text(now.day) + '_' + \
            get_text(now.hour) + get_text(now.minute) + get_text(now.second)


def format_time(seconds, with_ms=False):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    if days > 0:
        f += str(days) + '/'
    if hours > 0:
        f += str(hours) + ':'
    f += str(minutes) + '.' + str(secondsf)
    if with_ms and millis > 0:
        f += '_' + str(millis)
    return f


def load_file(filename):
    import os, json
    if not os.path.isfile(filename):
        print('文件不存在 "%s"' % filename)
        return None
    with open(filename) as f:
        return f.read()


def save_file(filename, text):
    with open(filename, 'w') as f:
        return f.write(text)


def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


# progress_bar

"""
import os
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
import time
last_time = time.time()
begin_time = last_time
last_len = -1
def progress_bar(current, total, msg=None):
    global last_time, begin_time, last_len
    if current == 0:
        begin_time = time.time()  # Reset for new bar.
        last_len = -1

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    cur_time = time.time()
    if last_len == cur_len and current < total - 1 and cur_time - last_time < 1 and msg is None:
        return

    import sys
    sys.stderr.write(' [')
    for i in range(cur_len):
        sys.stderr.write('=')
    sys.stderr.write('>')
    for i in range(rest_len):
        sys.stderr.write('.')
    sys.stderr.write(']')

    last_time = cur_time
    tot_time = cur_time - begin_time
    last_len = cur_len

    L = []
    est_time = tot_time / (current + 1) * total
    L.append(' Time:%s/Est:%s' % (format_time(tot_time), format_time(est_time)))
    if msg:
        L.append(' ' + msg)

    msg = ''.join(L)
    sys.stderr.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stderr.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stderr.write('\b')
    sys.stderr.write(' %d/%d ' % (current+1, total))

    if current < total - 1:
        sys.stderr.write('\r')
    else:
        sys.stderr.write('\n')
    sys.stderr.flush()

"""

