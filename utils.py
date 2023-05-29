import shutil
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import logging

class AccuracyCounter:
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0
        
    def addOntBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal
    
    def reportAccuracy(self):
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)

def cal_acc(gt_list, predict_list, num, writer, epoch, name):
    acc_sum = 0
    accu_set = {}
    for n in range(num):
        y = []
        pred_y = []
        for i in range(len(gt_list)):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt)
                pred_y.append(predict)
        print ('{}: {:4f} {}/{}'.format(n if n != (num - 1) else 'Unk', accuracy_score(y, pred_y), round(accuracy_score(y, pred_y) * len(y)), len(y)))
        if n != (num - 1):
            accu_set[name[n]] = accuracy_score(y, pred_y)
        if n == (num - 1):
            print ('OS*: {:4f}'.format(acc_sum / (num - 1)))
            OS_star = acc_sum / (num - 1)
            unk = accuracy_score(y, pred_y)
            writer.add_scalar('Known_ave', acc_sum / (num - 1), epoch)
            writer.add_scalar('Unknown', accuracy_score(y, pred_y), epoch)
        acc_sum += accuracy_score(y, pred_y)
    writer.add_scalars('Known_class', accu_set, epoch)
    print ('OS: {:4f}'.format(acc_sum / num))
    writer.add_scalar('Acc_ave', acc_sum / num, epoch)
    print ('Overall Acc : {:4f}'.format(accuracy_score(gt_list, predict_list)))
    writer.add_scalar('Over_all', accuracy_score(gt_list, predict_list), epoch)
    return  OS_star, unk

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

def list2tensor(list):
    return torch.cat(list, dim=0)

def process_dict(dict, points=3):
    for key in dict.keys():
        dict[key] = np.round(dict[key].item(), points)
    return dict

@torch.no_grad()
def ema_model_update(model,ema_model,ema_m):
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval*ema_m + param_train.detach()*(1 - ema_m))
    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def enable_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def opt_step(opt_list):
    for opt in opt_list:
        opt.step()
    for opt in opt_list:
        opt.zero_grad()

def print_and_log(data):
    print(data)
    logging.info(data)


