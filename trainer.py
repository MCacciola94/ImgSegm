import numpy as np
from utils.utils import *
import time
import os
import torch

class Trainer():

    def __init__(self, model, dataset, criterion, optimizer, lr_scheduler, save_dir,  print_freq):
        self.model = model
        self.dataset = dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.best_score = 0

        # Check the save_dir exists or not
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.print_freq = print_freq
        


    def train(self, epochs):
        # Starting time
        start=time.time()        
  
        for epoch in range(epochs):

            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            
            self.run_epoch(epoch)

            self.lr_scheduler.step()

            # evaluate on validation set
            d_score = self.validate()

            # remember best score and save checkpoint
            is_best = d_score> self.best_score
            self.best_score = max(d_score, self.best_score)
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_score': self.best_score,
                }, is_best, filename=os.path.join(self.save_dir, 'checkpoint_best.th'))

        print("\n Elapsed time for training ", elapsed_time(start))
        print("Best accuracy: ", self.best_score)





    def run_epoch(self, epoch):
        """
            Run one train epoch
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()


        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (img, target) in enumerate(self.dataset["train_loader"]):

            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda()
            img = img.cuda()


            # compute output
            output = self.model(img)
            
            loss = self.criterion(output, target)
            

            # compute gradient and do step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss= loss.item()
            output = output.float()


            # measure dice_score and record loss

            d_score = bd_score(output.data, target)

            losses.update(loss, img.size(0))

            top1.update(d_score, img.size(0))



            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f}) \t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(self.dataset["train_loader"]), batch_time = batch_time,
                        data_time = data_time, loss = losses, top1 = top1))
        


    def validate(self, reg_on = True):
        """
        Run evaluation
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            for i, (img, target) in enumerate(self.dataset["valid_loader"]):

                target = target.cuda()
                img = img.cuda()
                target = target.cuda()

                # compute output
                output = self.model(img)
                loss = self.criterion(output, target)
                loss = loss.item()
                output = output.float()


                # measure accuracy and record loss

                d_score = bd_score(output.data, target)

                losses.update(loss, img.size(0))

                top1.update(d_score, img.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                if i % self.print_freq == 0:

                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i, len(self.dataset["valid_loader"]), batch_time = batch_time, loss = losses,
                            top1 = top1))


        print(' * Prec@1 {top1.avg:.3f}'
            .format(top1 = top1))


        return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

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

def bd_score(output, target):
    """Computes the dice score for the batch"""
    with torch.no_grad():
        batch_size = target.size(0)
        res=0
        output=torch.sigmoid(output).round()

        output= output.cpu()
        target=target.cpu()

        for i in range(batch_size):
            mask= np.array(target[i])
            out = np.array(output[i])
            res += dice_score(out,mask)
        
        return res*100/batch_size


