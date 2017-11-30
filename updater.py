#!/usr/bin/env python

# -*- coding:utf-8 -*-

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np
import numba
import resource

def print_memory():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    print(ru.ru_maxrss)


class CompGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, tc=9000, td=1000, init_iter=0, alpha=0.0004, *args, **kwargs):
        self.gen, self.dis = kwargs.pop("models")
        self.tc, self.td = tc, td
        self.iteration = init_iter
        self.alpha = alpha
        super(CompGANUpdater, self).__init__(*args, **kwargs)

    def mse_gen(self, x, m, c):
        return F.sum(F.batch_l2_norm_squared(F.broadcast_to(m, x.shape) * (c - x))) / len(x)

    @numba.jit
    def make_mask(self, batchsize, x_h, x_w, hole_size, xp):
        mask = xp.zeros((batchsize, x_h, x_w))
        mask_random = xp.random.randint(0, hole_size + 1, (batchsize, 2))
        for i in range(batchsize):
            mask[i][mask_random[i, 0]:mask_random[i, 0] + hole_size, mask_random[i, 1]:mask_random[i, 1] + hole_size] = 1
        return mask[:, xp.newaxis, :, :].astype(xp.float32), mask_random

    def make_input_x(self, x, mask, xp):
        x_fill = F.mean(x, axis=(2, 3))[:, :, xp.newaxis, xp.newaxis]
        x_shape = x.shape
        return x * F.broadcast_to((1 - mask), x_shape) + F.broadcast_to(x_fill, x_shape) * F.broadcast_to(mask, x_shape)

    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_gen(self, gen, y_fake, mse):
        batchsize = len(y_fake)
        loss = mse + self.alpha * F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer("gen")
        dis_optimizer = self.get_optimizer("dis")

        batch = self.get_iterator("main").next()
        x = Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x.data)
        B, C, H, W = x.shape
        m, mr = self.make_mask(B, H, W, H//2, xp)
        m = Variable(self.converter(m, self.device))
        
        gen, dis = self.gen, self.dis

        c = gen(self.make_input_x(x, m, xp), m)
        m = m.data

        if self.iteration < self.tc:
            chainer.report({'loss': self.mse_gen(x.data, m, c)}, gen)
            # x.data, m.data is OK?
            gen_optimizer.update(self.mse_gen, x.data, m, c)
        else:
            _, mrd = self.make_mask(B, H, W, H//2, xp)
            y_real = dis(x, mrd)
            y_fake = dis(c, mr)
            dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
            if self.iteration > self.tc + self.td:
                gen_optimizer.update(self.loss_gen, gen, y_fake, self.mse_gen(x.data, m, c))

        self.iteration += 1


def test():
    print(CompGANUpdater.make_mask(None, 2, 4, 4, 2))
    x = np.random.randint(1, 4, (2, 3, 4, 4))
    m = CompGANUpdater.make_mask(None, 2, 4, 4, 2)[0]
    c = np.ones((2, 3, 4, 4))
    print(m.shape)
    print(m * (c - x))
    print(CompGANUpdater.make_input_x(None, x, m))
    print(x[x*m==0])

if __name__ == "__main__":
    test()

