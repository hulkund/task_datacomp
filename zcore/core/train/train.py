import IPython
import math
import os
import random
import sys
import time
import torch
from .train_utils import *

# Based on https://github.com/zhangxin-xd/Dataset-Pruning-TDDS/blob/main/train_subset.py

def train_coreset_model(args, model, train_loader, test_loader):

    # Log setup.
    if args.results_dir == None: 
        args.results_dir = os.path.join(
            os.path.dirname(args.score_file), 
            f"p{int(args.prune_rate*100)}-s{args.manual_seed}",
        )
    if not os.path.isdir(args.results_dir): os.makedirs(args.results_dir)
    log = open(os.path.join(args.results_dir, "train_log.txt"), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(f"Train settings : {state}", log)
    print_log(f"Train model :\n{model}", log)
    print_log(f"python version : {sys.version.replace(chr(10), ' ')}", log)
    print_log(f"torch version : {torch.__version__}", log)
    print_log(f"cudnn version : {torch.backends.cudnn.version()}", log)

    # Model setup.
    model = torch.nn.DataParallel(model, device_ids=[args.device])
    model.to(args.device)

    # Train setup.
    optimizer = torch.optim.SGD(
        model.parameters(), 
        state['learning_rate'], 
        momentum=state['momentum'],
        weight_decay=state['decay'], 
        nesterov=True
    )
    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer = optimizer,
        T_max =  args.epochs * len(train_loader)
    )
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(args.device)
    recorder = RecorderMeter(args)
    epoch_time = AverageMeter()

    # Train loop.
    for epoch in range(args.epochs):
        tic = time.time()
        epoch_header(args, scheduler, recorder, epoch_time, epoch, log)

        # Train and validation.
        train_acc, train_loss = train(args, model, train_loader, optimizer,
                                      scheduler, criterion, epoch, log)
        val_acc, val_loss = validate(args, model, test_loader, criterion, log)

        # Update checkpoint and log.
        is_best = recorder.update(epoch, train_acc, train_loss, val_acc, 
                                  val_loss)
        save_checkpoint(args, epoch, model, recorder, optimizer, is_best)
        recorder.plot_curve()
        epoch_time.update(time.time() - tic)

    epoch_header(args, scheduler, recorder, epoch_time, epoch+1, log)
    log.close()
    print(f"\nModel training on coreset complete. Log saved at {args.results_dir}")

def epoch_header(args, scheduler, recorder, epoch_time, epoch, log):
    current_lr = scheduler.get_last_lr()[0]
    hours, mins, secs = secs2time(epoch_time.avg * (args.epochs - epoch))
    print_log(
        f"\n{time_string():s} [Epoch={epoch:03d}/{args.epochs:03d}] "\
        f"[Need : {hours:02d}:{mins:02d}:{secs:02d}] [learning_rate="\
        f"{scheduler.get_last_lr()[0]:6.4f}] [Best : Accuracy="\
        f"{recorder.max_accuracy(False):.2f}, Error="\
        f"{(100-recorder.max_accuracy(False)):.2f}]", log
    )

def train(args, model, train_loader, optimizer, scheduler, criterion, epoch, log):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    for t, (model_input, target) in enumerate(train_loader):
        tic = time.time()

        x = torch.autograd.Variable(model_input.to(args.device))
        y = torch.autograd.Variable(target[0].to(args.device))
        w = target[1].to(args.device)

        output = model(x)
        loss = criterion(output, y) * torch.mean(w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        prec1, prec5 = topk_accuracy(output.data, y, topk=(1, 5))
        losses.update(loss.item(), len(y))
        top1.update(prec1.item(), len(y))
        top5.update(prec5.item(), len(y))

        batch_time.update(time.time() - tic)
        if t % args.print_freq == 0:
            print_log(f" Epoch: [{epoch:03d}][{t:03d}/{args.batch_size:03d}] "\
                      f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "\
                      f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "\
                      f"Loss {losses.val:.4f} ({losses.avg:.4f}) "\
                      f"Prec@1 {top1.val:.3f} ({top1.avg:.3f}) "\
                      f"Prec@5 {top5.val:.3f} ({top5.avg:.3f}) "\
                      f"{time_string()}", log
            )

    print_log(f"  Train Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 "\
              f"{(100-top1.avg):.3f}", log)

    return top1.avg, losses.avg
                                      
def validate(args, model, test_loader, criterion, log):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (model_input, target) in enumerate(test_loader):
            x = model_input.to(args.device)
            y = target.to(args.device)

            output = model(x)
            loss = criterion(output, y)
            
            prec1, prec5 = topk_accuracy(output.data, y, topk=(1, 5))
            losses.update(loss.item(), len(y))
            top1.update(prec1.item(), len(y))
            top5.update(prec5.item(), len(y))

    print_log(f"  Test  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 "\
              f"{(100-top1.avg):.3f}", log)

    return top1.avg, losses.avg
