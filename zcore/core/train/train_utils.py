import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import torch

# Based on https://github.com/zhangxin-xd/Dataset-Pruning-TDDS/blob/main/utils.py 

def print_log(print_string, log):
    print(f"{print_string}")
    log.write(f"{print_string}\n")
    log.flush() 

class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, args):
    self.reset(args.epochs)
    self.title = f"Loss/Accuracy for {args.dataset} "\
                 f"p{int(args.prune_rate*100)} Train/Val"
    self.figure_file = os.path.join(args.results_dir, "train_val_plot.png")

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_acc, train_loss, val_acc, val_loss):
    assert idx >= 0 and idx < self.total_epoch, f"total_epoch : {self.total_epoch} , but update with the {idx} index"
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()
  
  def plot_curve(self):
    dpi = 150  
    width, height = 1200, 800
    legend_fontsize = 13
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 20
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(self.title, fontsize=20)
    plt.xlabel("training epoch", fontsize=16)
    plt.ylabel("accuracy", fontsize=16)
  
    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='b', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='r', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)
    
    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='b', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if self.figure_file is not None:
      fig.savefig(self.figure_file, dpi=dpi, bbox_inches='tight')
      print (f"-- saved figure {self.title} into {self.figure_file}")
    plt.close(fig)

class AverageMeter(object):
  """Computes and stores the average and current value"""
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

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()+28800) ))
    return string

def secs2time(epoch_time):
    hours = int(epoch_time / 3600)
    mins = int((epoch_time - 3600*hours) / 60)
    secs = int(epoch_time - 3600*hours - 60*mins)
    return hours, mins, secs

def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    accuracy = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accuracy.append(correct_k.mul_(100.0 / batch_size))
    return accuracy

def save_checkpoint(args, epoch, model, recorder, optimizer, is_best):
    state = {"epoch": epoch+1, "arch": args.architecture, "state_dict": model,
             "recorder": recorder, "optimizer": optimizer.state_dict()}
    filename = os.path.join(args.results_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(args.results_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)
