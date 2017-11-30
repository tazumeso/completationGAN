#!/usr/bin/env python

# -*- coding:utf-8 -*-

import os

import numpy as np
import numba
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import chainer
import chainer.cuda
import chainer.functions as F
from chainer import Variable


@numba.jit
def make_mask(batchsize, x_h, x_w, hole_size):
    mask = np.zeros((batchsize, x_h, x_w))
    mask_random = np.random.randint(0, hole_size + 1, (batchsize, 2))
    for i in range(batchsize):
        mask[i][mask_random[i, 0]:mask_random[i, 0] + hole_size, mask_random[i, 1]:mask_random[i, 1] + hole_size] = 1
    return mask[:, np.newaxis, :, :].astype(np.float32), mask_random


def make_input_x(x, mask):
    x_fill = F.mean(x, axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    x_shape = x.shape
    return x * F.broadcast_to((1 - mask), x_shape) + F.broadcast_to(x_fill, x_shape) * F.broadcast_to(mask, x_shape)


def out_generated_image(gen, test, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        m, mr = make_mask(16, 128, 128, 64)
        m = Variable(xp.array(m))
        t = Variable(xp.array(test))
        with chainer.using_config("train", False):
            x = gen(make_input_x(t, m), m)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        preview_dir = "{}/preview".format(dst)
        preview_path = preview_dir + "/image{:0>8}.png".format(trainer.updater.iteration)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        x = np.asarray(np.clip(x * 255, 0., 255.), dtype=np.uint8)
        plt.figure(figsize=(10, 10))
        for i in range(n_images):
            plt.subplot2grid((rows, cols), (i // cols, i % cols), colspan=1, rowspan=1)
            plt.imshow(x[i].reshape(3, 128, 128).transpose(1,2,0))
            plt.axis("off")
        plt.savefig(preview_path)
        plt.close()

    return make_image


def initial_image(test, rows, cols, seed, dst):
    np.random.seed(seed)
    n_images = rows * cols
    m, mr = make_mask(16, 128, 128, 64)
    m = Variable(np.array(m))
    t = Variable(np.array(test))
    x = make_input_x(t, m)
    np.random.seed()

    preview_dir = "{}/preview".format(dst)
    preview_path = preview_dir + "/image_start.png"
    test_path = preview_dir + "/image_ground_trueth.png"
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
        
    x = chainer.cuda.to_cpu(x.data)
    t = chainer.cuda.to_cpu(t.data)

    x = np.asarray(np.clip(x * 255, 0., 255.), dtype=np.uint8)
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        plt.subplot2grid((rows, cols), (i // cols, i % cols), colspan=1, rowspan=1)
        plt.imshow(x[i].reshape(3, 128, 128).transpose(1,2,0))
        plt.axis("off")
    plt.savefig(preview_path)
    plt.close()

    t = np.asarray(np.clip(t * 255, 0., 255.), dtype=np.uint8)
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        plt.subplot2grid((rows, cols), (i // cols, i % cols), colspan=1, rowspan=1)
        plt.imshow(t[i].reshape(3, 128, 128).transpose(1,2,0))
        plt.axis("off")
    plt.savefig(test_path)
    plt.close()