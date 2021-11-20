from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.distributed as dist
import torch.backends.cudnn as cudnn
# from visdom import Visdom
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import pdb


from data.config import cfg
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
from data.factory import dataset_factory, detection_collate
from models.s3fd import build_s3fd,build_s3fd_repvgg
from utils import csv_write



def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='S3FD face Detector Training With Pytorch')
parser.add_argument('--dataset',default='face',
                    choices=['hand', 'face', 'head'],
                    help='Train target')
parser.add_argument('--basenet',default=cfg.pritrian_pth,
                    help='Pretrained base model')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size',default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--pretrained', default=False, type=str2bool,
                    help='use pre-trained model')
parser.add_argument('--resume',default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--distributed', default=True, type=str,
                    help='use distribute training')
parser.add_argument("--local_rank", default=0, type=int)                  
parser.add_argument('--lr', '--learning-rate',default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save_folder',default='/content/drive/MyDrive/repos/S3FD_RepVGG/weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--cuda',
                    default=True, type=str2bool,
                    help='Use CUDA to train model')
args = parser.parse_args()

# plt.figure(figsize =(200,80),dpi =80 )
# x =range(20)
# y = torch.randn(20)
# plt.plot(x,y)
# plt.show()
# viz = Visdom()
# os.system('chdir /content/drive/MyDrive/S3FD_RepVGG')
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cudnn.benchmark = True
args = parser.parse_args()
minmum_loss = np.inf

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def main():
    global args
    global minmum_loss
    
    global writer
    root = '/content/drive/MyDrive/repos/S3FD_RepVGG'
    log_path = 'tersorborard_log'
    log_path = os.path.join(root,log_path)
    if not os.path.exists(log_path):
      os.mkdir(log_path)

    writer =SummaryWriter(log_path,comment='Repvgg_se_train_loss')
    # viz.line([[0.0,0.0]],[0.0],win = 'Train_loss',
    #         opts = dict(xlabel='iter_num',ylabel= 'Train_loss',title ='Train_loss'))
    # viz.line([[0.0, 0.0]], [0.0], win='Val_loss',
    #          opts=dict(xlabel='iter_num', ylabel='Val_loss', title='Val_loss'))
    # if os.path.exists('train_loss.csv'):
    #     os.system('rm train_loss.csv')
        
    # csv_write('train_loss.csv',cfg.TRAIN_LOSS_TILE)


    # if os.path.exists('val_loss.csv'):
    #     os.system('rm val_loss.csv')
        
    # csv_write('/content/drive/MyDrive/repos/S3FD_RepVGG/val_loss.csv',cfg.VAL_LOSS_TILE)

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                                init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    # build S3FD 
    print("Building net...")
    # s3fd_net = build_s3fd('train', cfg.NUM_CLASSES)
    s3fd_net = build_s3fd_repvgg('train', cfg.NUM_CLASSES)
    model = s3fd_net
    print(model)
    # if args.pretrained:
    #     vgg_weights = torch.load(args.save_folder + args.basenet)
    #     print('Load base network....')
    #     model.vgg.load_state_dict(vgg_weights)
    if args.pretrained:
        # pdb.set_trace()
        # checkpoint = torch.load(os.path.join(args.save_folder,args.basenet))
        checkpoint = torch.load('/content/drive/MyDrive/repos/S3FD_RepVGG/weights/RepVGG-A1-Train.pth')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        ckpt = {k.replace('stage', ''): v for k, v in checkpoint.items()} 
        

        del ckpt['linear.weight']
        del ckpt['linear.bias']
        # strip the names
        model.Repvgg.load_state_dict(ckpt)
        


    # for multi gpu
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)

    model = model.cuda()
    # optimizer and loss function  
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    # criterion = MultiBoxLoss(cfg, args.dataset,True)
    criterion = MultiBoxLoss(cfg = cfg,overlap_thresh = cfg.FACE.OVERLAP_THRESH,
                            prior_for_matching = True,bkg_label = 0,
                            neg_mining = True, neg_pos = 3,neg_overlap = 0.5,
                            encode_target = False, use_gpu = args.cuda,loss_name = cfg['losstype'])
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
            args.start_epoch = checkpoint['epoch']
            minmum_loss = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print('Initializing weights...')
        model.Repvgg.apply(s3fd_net.weights_init)
        model.extras.apply(s3fd_net.weights_init)
        model.loc.apply(s3fd_net.weights_init)
        model.conf.apply(s3fd_net.weights_init)

    print('Loading wider dataset...')
    train_dataset, val_dataset = dataset_factory(args.dataset)

    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                collate_fn=detection_collate,
                                pin_memory=True)

    val_batchsize = args.batch_size // 2
    val_loader = data.DataLoader(val_dataset, val_batchsize,
                                num_workers=args.num_workers,
                                shuffle=False,
                                collate_fn=detection_collate,
                                pin_memory=True)

    print('Using the specified args:')
    print(args)
    # load PriorBox
    # priorbox = PriorBox(cfg)
    # with torch.no_grad():
    #     priors = priorbox.forward()
    #     priors = priors.cuda()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        end = time.time()
        train_loss = train(train_loader, model,  criterion, optimizer, epoch)
        val_loss = val(val_loader, model, criterion,epoch)
        if args.local_rank == 0:
            is_best = val_loss < minmum_loss
            minmum_loss = min(val_loss, minmum_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': minmum_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch)
        epoch_time = time.time() -end
        print('Epoch %s time cost %f' %(epoch, epoch_time))


def train(train_loader, model,  criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_loss = AverageMeter()
    cls_loss = AverageMeter()
 
    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader, 1):
        input, targets = data
        train_loader_len = len(train_loader)

        lr = adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)
        if args.cuda:
            input_var = Variable(input.cuda())
            target_var = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]

                
        else:
            input_var = Variable(input)
            target_var = [Variable(ann, volatile=True) for ann in targets]

        # input_var = Variable(input.cuda())
        # target_var = [Variable(ann.cuda(), requires_grad=False) for ann in targets]

        # compute output
        output = model(input_var)
       
        loss_l, loss_c = criterion(output, target_var)
        loss = loss_l + loss_c

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            reduced_loss_l = reduce_tensor(loss_l.data)
            reduced_loss_c = reduce_tensor(loss_c.data)
        else:
            reduced_loss = loss.data
            reduced_loss_l = loss_l.data
            reduced_loss_c = loss_c.data
        losses.update(to_python_float(reduced_loss), input.size(0))
        loc_loss.update(to_python_float(reduced_loss_l), input.size(0))
        cls_loss.update(to_python_float(reduced_loss_c), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i >= 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {3:.3f} ({4:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'loc_loss {loc_loss.val:.3f} ({loc_loss.avg:.3f})\t'
                  'cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})'.format(
                   epoch, i, train_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, loc_loss=loc_loss, cls_loss=cls_loss))
            # viz_update(step_size,loss,avg_loss,iter_num,epoch_num,win)
            # viz_update(train_loader_len,losses.val,losses.avg,i,epoch,'Train_loss')
            # print_datas = [epoch,i,losses.val,losses.avg,loc_loss.val,loc_loss.avg,cls_loss.val,cls_loss.avg,lr]
            writer.add_scalar('train_loss/loss_val',losses.val,epoch*train_loader_len+i)
            writer.add_scalar('train_loss/loss_avg',losses.avg,epoch*train_loader_len+i)
            writer.add_scalar('lr',lr,epoch*train_loader_len+i)
            # csv_write('train_loss.csv',print_datas)
    return losses.avg


def val(val_loader, model,  criterion,epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_loss = AverageMeter()
    cls_loss = AverageMeter()

    # switch to train mode
    model.eval()
    end = time.time()

    for i, data in enumerate(val_loader, 1):
        input, targets = data
        val_loader_len = len(val_loader)

        # measure data loading time
        data_time.update(time.time() - end)

        input_var = Variable(input.cuda())
        target_var = [Variable(ann.cuda(), requires_grad=False) for ann in targets]

        # compute output
        output = model(input_var)
        loss_l, loss_c = criterion(output, target_var)
        loss = loss_l + loss_c

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            reduced_loss_l = reduce_tensor(loss_l.data)
            reduced_loss_c = reduce_tensor(loss_c.data)
        else:
            reduced_loss = loss.data
            reduced_loss_l = loss_l.data
            reduced_loss_c = loss_c.data
        losses.update(to_python_float(reduced_loss), input.size(0))
        loc_loss.update(to_python_float(reduced_loss_l), input.size(0))
        cls_loss.update(to_python_float(reduced_loss_c), input.size(0))

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0 and i >= 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Val_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Val_loc_loss {loc_loss.val:.3f} ({loc_loss.avg:.3f})\t'
                  'Val_cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})'.format(
                   i, val_loader_len,
                   args.total_batch_size / batch_time.val,
                   args.total_batch_size / batch_time.avg,
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, loc_loss=loc_loss, cls_loss=cls_loss))
            # viz_update(val_loader_len, losses.val, losses.avg, i, epoch, 'Val_loss')
            # print_datas = [epoch,i,losses.val,losses.avg,loc_loss.val,loc_loss.avg,cls_loss.val,cls_loss.avg]
                   
            # csv_write('val_loss.csv',print_datas)
            writer.add_scalar('val_loss/loss_val',losses.val,epoch*val_loader_len+i)
            writer.add_scalar('val_loss/loss_avg',losses.avg,epoch*val_loader_len+i)
            # writer.add_scalar('lr',lr,epoch*train_loader_len+i)
    return losses.avg


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

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

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 10

    if epoch >= 30:
        factor = factor + 1

    lr = args.lr * (0.1 ** factor)

    """Warmup"""
    if epoch < 1:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(args.local_rank == 0 and step % args.print_freq == 0 and step > 1):
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(state, is_best, epoch):
    filename = os.path.join(args.save_folder, "S3FD_" + str(epoch)+ ".pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_folder, 'model_best.pth'))
# def viz_update(step_size,loss,avg_loss,iter_num,epoch_num,win):
#     viz.line([[loss,avg_loss]],[epoch_num*step_size+iter_num],win= win,update = 'append')

if __name__ == '__main__':
    main()
    writer.close()