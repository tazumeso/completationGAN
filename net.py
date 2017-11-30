#!/usr/bin/env python

# -*- coding:utf-8 -*-


import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import cuda
import numba

import numpy as np


class Generator(chainer.Chain):
    def __init__(self, input_width=256, input_height=256, ch=256, wscale=0.2):
        super(Generator, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.ch = ch

        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            # O = (I + 2P - F) / S + 1
            self.c0 = L.Convolution2D(4, ch//4, 5, 1, 2, initialW=w)
            # change karnel size from original
            self.c1 = L.Convolution2D(ch//4, ch//2, 4, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(ch//2, ch//2, 3, 1, 1, initialW=w)
            # same as up
            self.c3 = L.Convolution2D(ch//2, ch, 4, 2, 1, initialW=w)
            self.c4 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c5 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            # dilated convolution
            # O = {I + 2P - F - (F-1) * (D-1)} / S + 1
            self.c6 = L.DilatedConvolution2D(ch, ch, 3, 1, 2, 2, initialW=w)
            self.c7 = L.DilatedConvolution2D(ch, ch, 3, 1, 4, 4, initialW=w)
            self.c8 = L.DilatedConvolution2D(ch, ch, 3, 1, 8, 8, initialW=w)
            self.c9 = L.DilatedConvolution2D(ch, ch, 3, 1, 16, 16, initialW=w)
            self.c10 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c11 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            # deconv
            # O = S * (I-1) + F - 2P
            self.c12 = L.Deconvolution2D(ch, ch//2, 4, 2, 1, initialW=w)
            self.c13 = L.Convolution2D(ch//2, ch//2, 3, 1, 1, initialW=w)
            self.c14 = L.Deconvolution2D(ch//2, ch//4, 4, 2, 1, initialW=w)
            self.c15 = L.Convolution2D(ch//4, ch//8, 3, 1, 1, initialW=w)
            self.c16 = L.Convolution2D(ch//8, 3, 3, 1, 1, initialW=w)
            # BN
            self.bn0 = L.BatchNormalization(ch//4)
            self.bn1 = L.BatchNormalization(ch//2)
            self.bn2 = L.BatchNormalization(ch//2)
            self.bn3 = L.BatchNormalization(ch)
            self.bn4 = L.BatchNormalization(ch)
            self.bn5 = L.BatchNormalization(ch)
            self.bn6 = L.BatchNormalization(ch)
            self.bn7 = L.BatchNormalization(ch)
            self.bn8 = L.BatchNormalization(ch)
            self.bn9 = L.BatchNormalization(ch)
            self.bn10 = L.BatchNormalization(ch)
            self.bn11 = L.BatchNormalization(ch)
            self.bn12 = L.BatchNormalization(ch//2)
            self.bn13 = L.BatchNormalization(ch // 2)
            self.bn14 = L.BatchNormalization(ch // 4)
            self.bn15 = L.BatchNormalization(ch // 8)

    def __call__(self, x, m):
        h = F.relu(self.bn0(self.c0(F.concat((x, m), axis=1))))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h)))
        h = F.relu(self.bn3(self.c3(h)))
        h = F.relu(self.bn4(self.c4(h)))
        h = F.relu(self.bn5(self.c5(h)))
        h = F.relu(self.bn6(self.c6(h)))
        h = F.relu(self.bn7(self.c7(h)))
        h = F.relu(self.bn8(self.c8(h)))
        h = F.relu(self.bn9(self.c9(h)))
        h = F.relu(self.bn10(self.c10(h)))
        h = F.relu(self.bn11(self.c11(h)))
        h = F.relu(self.bn12(self.c12(h)))
        h = F.relu(self.bn13(self.c13(h)))
        h = F.relu(self.bn14(self.c14(h)))
        h = F.relu(self.bn15(self.c15(h)))
        h = F.sigmoid(self.c16(h))
        h = x * F.broadcast_to((1 - m), x.shape) + h * F.broadcast_to(m, x.shape)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, input_width=256, input_height=256, ch=512, wscale=0.2):
        w = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.ch = ch
        with self.init_scope():
            self.cg0 = L.Convolution2D(3, ch//8, 4, 2, 1, initialW=w)
            self.cg1 = L.Convolution2D(ch//8, ch//4, 4, 2, 1, initialW=w)
            self.cg2 = L.Convolution2D(ch//4, ch//2, 4, 2, 1, initialW=w)
            self.cg3 = L.Convolution2D(ch//2, ch, 4, 2, 1, initialW=w)
            self.cg4 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)
            self.cg5 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)
            # should change use_gamma=False?
            self.bng0 = L.BatchNormalization(ch//8)
            self.bng1 = L.BatchNormalization(ch//4)
            self.bng2 = L.BatchNormalization(ch//2)
            self.bng3 = L.BatchNormalization(ch)
            self.bng4 = L.BatchNormalization(ch)
            self.bng5 = L.BatchNormalization(ch)
            self.lg = L.Linear(4 * 4 * 512, 1024, initialW=w)

            self.cl0 = L.Convolution2D(3, ch//8, 4, 2, 1, initialW=w)
            self.cl1 = L.Convolution2D(ch//8, ch//4, 4, 2, 1, initialW=w)
            self.cl2 = L.Convolution2D(ch//4, ch//2, 4, 2, 1, initialW=w)
            self.cl3 = L.Convolution2D(ch//2, ch, 4, 2, 1, initialW=w)
            self.cl4 = L.Convolution2D(ch, ch, 4, 2, 1, initialW=w)
            self.bnl0 = L.BatchNormalization(ch // 8)
            self.bnl1 = L.BatchNormalization(ch // 4)
            self.bnl2 = L.BatchNormalization(ch // 2)
            self.bnl3 = L.BatchNormalization(ch)
            self.bnl4 = L.BatchNormalization(ch)
            self.ll = L.Linear(4 * 4 * 512, 1024, initialW=w)

            self.lo = L.Linear(2048, 1, initialW=w)

    def __call__(self, x, mr):
        # xc is complitation resion
        xc = self.make_xc(x.data, mr)
        hg = F.relu(self.bng0(self.cg0(x)))
        hg = F.relu(self.bng1(self.cg1(hg)))
        hg = F.relu(self.bng2(self.cg2(hg)))
        hg = F.relu(self.bng3(self.cg3(hg)))
        hg = F.relu(self.bng4(self.cg4(hg)))
        #hg = F.relu(self.bng5(self.cg5(hg)))
        hg = F.relu(self.lg(hg))

        hl = F.relu(self.bnl0(self.cl0(xc)))
        hl = F.relu(self.bnl1(self.cl1(hl)))
        hl = F.relu(self.bnl2(self.cl2(hl)))
        hl = F.relu(self.bnl3(self.cl3(hl)))
        #hl = F.relu(self.bnl4(self.cl4(hl)))
        hl = F.relu(self.ll(hl))

        h = F.concat((hg, hl), axis=1)
        h = self.lo(h)
        return h
    
    def make_xc(self, x, mr):
        B, C, H, W = x.shape
        xp = chainer.cuda.get_array_module(x)
        xc = xp.zeros((B, C, H//2, W//2), dtype=xp.float32)
        for i in range(B):
            xc[i] = x[i][:, mr[i, 0]:mr[i, 0] + H//2, mr[i, 1]:mr[i, 1] + W//2]
        return Variable(xc.astype(xp.float32))




