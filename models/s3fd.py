#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
# from _typeshed import NoneType

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from layers.functions import Detect
from layers.modules import L2Norm
from data.config import cfg
from layers.functions.prior_box import PriorBox
import numpy as np
# from RepVgg.repvgg import get_RepVGG_func_by_name
from RepVgg.repvgg_dcv import get_RepVGG_func_by_name

class S3FD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes):
        super(S3FD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        '''
        self.priorbox = PriorBox(size,cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        '''
        # SSD network
        # self.vgg = nn.ModuleList(base)
        self.Repvgg = nn.ModuleList(base) #change base to Repvgg
        # Layer learns to scale the l2 normalized features from conv4_3
        #change the output channel
        self.L2Norm3_3 = L2Norm(int(64*cfg.width_multiplier[0]), 10)
        self.L2Norm4_3 = L2Norm(int(128*cfg.width_multiplier[1]), 8)
        self.L2Norm5_3 = L2Norm(int(256*cfg.width_multiplier[2]), 5)

        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)
 

    def forward_ssd(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv3_3 relu
        for k in range(16):
            x = self.vgg[k](x)
            # print(f"layer {k}",x.shape)
            
        s = self.L2Norm3_3(x)
        # print('extrac layer 1',x.shape)
        sources.append(s)

        # apply vgg up to conv4_3 relu
        for k in range(16, 23):
            x = self.vgg[k](x)
            # print(f"layer {k}",x.shape)
        s = self.L2Norm4_3(x)
        # print('extrac layer 2',x.shape)
        sources.append(s)

        # apply vgg up to conv5_3 relu
        for k in range(23, 30):
            x = self.vgg[k](x)
            # print(f"layer {k}",x.shape)
        s = self.L2Norm5_3(x)
        # print('extrac layer 3',x.shape)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        # s = self.L2Norm5_3(x)
            # print(f"layer {k}",x.shape)
        # print('extrac layer 4',x.shape)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                # print(f'extrac layer {k//2 + 5}',x.shape)
                sources.append(x)



        # 在 SSD 中， 直接将不同feature map的预测结果 append 起来，
        # 这里对 conv3_3 的特征图做特殊处理
        # 对于conv3_3层是需要使用max-out label的
        '''
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        '''
        # apply multibox head to source layers
        # conv_3_3
        loc_x = self.loc[0](sources[0])
        conf_x = self.conf[0](sources[0])

        #  取 top3 的置信度得分
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]
            
        self.priorbox = PriorBox(size,features_maps,cfg)
        with torch.no_grad():
                self.priors = self.priorbox.forward()
                
                
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
           
                
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),self.priors.type(type(x.data)))     # conf preds)
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),self.priors
            )
        return output
    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        size = x.size()[2:]
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv3_3 relu
        for k in range(2):
            x = self.Repvgg[k](x)
            # print(f"layer {k}",x.shape)
            
        s = self.L2Norm3_3(x)
        # print('extrac layer 1',x.shape)
        sources.append(s)
        x = self.Repvgg[2](x)
        # apply vgg up to conv4_3 relu
        # for k in range(16, 23):
        #     x = self.vgg[k](x)
        #     # print(f"layer {k}",x.shape)
        s = self.L2Norm4_3(x)
        # print('extrac layer 2',x.shape)
        sources.append(s)
        for i in range(6):
            x = self.Repvgg[3][i](x)

        # apply vgg up to conv5_3 relu
        # for k in range(23, 30):
        #     x = self.vgg[k](x)
        #     # print(f"layer {k}",x.shape)
        s = self.L2Norm5_3(x)
        # print('extrac layer 3',x.shape)
        sources.append(s)
        for i in range(14)[6:]:
            x = self.Repvgg[3][i](x)
        
        s = self.L2Norm5_3(x)
        sources.append(s)

        
        x = self.Repvgg[4](x)
        # apply vgg up to fc7
        # for k in range(30, len(self.vgg)):
        #     x = self.vgg[k](x)
            # print(f"layer {k}",x.shape)
        # print('extrac layer 4',x.shape)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                # print(f'extrac layer {k//2 + 5}',x.shape)
                sources.append(x)



        # 在 SSD 中， 直接将不同feature map的预测结果 append 起来，
        # 这里对 conv3_3 的特征图做特殊处理
        # 对于conv3_3层是需要使用max-out label的
        '''
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        '''
        # apply multibox head to source layers
        # conv_3_3
        for i in range(2):
            loc_x = self.loc[i](sources[i])
            conf_x = self.conf[i](sources[i])

            #  取 top3 的置信度得分
            max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
            conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

            loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
            conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(2, len(sources)):
            x = sources[i]
            conf.append(self.conf[i](x).permute(0, 2, 3, 1).contiguous())
            loc.append(self.loc[i](x).permute(0, 2, 3, 1).contiguous())

        features_maps = []
        for i in range(len(loc)):
            feat = []
            feat += [loc[i].size(1), loc[i].size(2)]
            features_maps += [feat]
            
        self.priorbox = PriorBox(size,features_maps,cfg)
        with torch.no_grad():
                self.priors = self.priorbox.forward()
                
                
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
           
                
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),self.priors.type(type(x.data)))     # conf preds)
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform_(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            if type(m.bias) !=type(None) :
                m.bias.data.zero_()


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512,'M']

extras_cfg = [256, 'S', 512, 128, 'S', 256]

Repvgg_name = cfg.RepVgg_Name
multiplier = cfg.width_multiplier

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [conv6,nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def RepVgg(name,deploy):
    repvgg_build_func = get_RepVGG_func_by_name(name)
    rep_vgg_model = repvgg_build_func(deploy)

    backbone = torch.nn.Sequential(rep_vgg_model.stage0, rep_vgg_model.stage1,rep_vgg_model.stage2,
    rep_vgg_model.stage3,rep_vgg_model.stage4)
    
    
    return backbone

    

def add_extras(cfg, i, batch_norm=False):
    """
    向VGG网络中添加额外的层用于feature scaling。
    extras_cfg = [256, 'S', 512, 128, 'S', 256]
    """
    layers = []
    in_channels = i#1024 for vgg/ 1280 for repvgg
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, num_classes):
    """
    conv3-3输出1对应face的分类，
    在conv3_3层(小目标产生最多的层)，输出维度为Ns+4，Ns = Nm + 1，其Nm对应maxout bg label，以去除小目标的误检；
    (本项目中，设置为  3 + (num_classes-1)=4 )。
    而其它所有检测层的输出通道数均为（2+4），表示二分类和4个bounding box坐标。 
    Ns中包含了1个正样本的概率以及Ns−1个负样本概率，我们从负样本概率中选出最大值与正样本概率一起完成之后的softmax二分类。 
    这种看似多余的操作实际上是通过提高分类难度来加强分类能力。
    """
    loc_layers = []
    conf_layers = []
    # vgg_source = [3,4]

    loc_layers += [nn.Conv2d(vgg[2][0].in_channels, 4,
                             kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(vgg[2][0].in_channels,
                              3 + (num_classes-1), kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(vgg[3][0].in_channels,
                                  3 + (num_classes-1), kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(vgg[3][0].in_channels,
                                   3 + (num_classes-1), kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(vgg[3][6].in_channels,
                                 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(vgg[3][6].in_channels,
                                  num_classes, kernel_size=3, padding=1)]
                                  
    loc_layers += [nn.Conv2d(vgg[4][0].in_channels,
                                 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(vgg[4][0].in_channels,
                                  num_classes, kernel_size=3, padding=1)]

    # for k, v in enumerate(vgg_source):
    #     loc_layers += [nn.Conv2d(vgg[v][0].in_channels,
    #                              4, kernel_size=3, padding=1)]
    #     conf_layers += [nn.Conv2d(vgg[v][0].in_channels,
    #                               num_classes, kernel_size=3, padding=1)]

    loc_layers += [nn.Conv2d(int(512*cfg.width_multiplier[3]), 4,
                             kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(int(512*cfg.width_multiplier[3]),
                              num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layers[1::2], 2):
        # 以步长为2在一个list列表中取 extra_layers 的 1，3,5层
        loc_layers += [nn.Conv2d(v.out_channels,
                                 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels,
                                  num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


def build_s3fd(phase, num_classes=2):
    base_, extras_, head_ = multibox(
        vgg(vgg_cfg, 3), add_extras((extras_cfg), 1024), num_classes)
    
    return S3FD(phase, base_, extras_, head_, num_classes)

def build_s3fd_repvgg(phase, num_classes=2,deploy =False):
    base_, extras_, head_ = multibox(
        RepVgg(cfg.RepVgg_Name, deploy), add_extras((extras_cfg), int(512*cfg.width_multiplier[3])), num_classes)
    
    return S3FD(phase, base_, extras_, head_, num_classes)
if __name__ == '__main__':
    net = build_s3fd_repvgg('train', num_classes=2)
    print(net)
    inputs = Variable(torch.randn(16, 3, 640, 640))
    
    output = net(inputs)

