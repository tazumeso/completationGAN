#!/usr/bin/env python

# -*- coding:utf-8 -*-

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda

from net import *
from updater import CompGANUpdater
from visualize import out_generated_image, initial_image
import numpy as np


def load_train():
    train = np.load("streetview/google_streetview.npy").astype(np.float32)[:, ::-1, :, :] / 255.
    return train[:242800], train[242800:]


def main():
    parser = argparse.ArgumentParser(description="Complitation GAN")
    parser.add_argument("--batchsize", "-b", type=int, default=50, help="Number of images in each mini-batch")
    parser.add_argument("--iter", "-i", type=int, default=1000000, help="Number of iter to learning")
    parser.add_argument("--gpu", "-g", type=int, default=-1, help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--out", "-o", default="result", help="Directory to output the result")
    parser.add_argument("--snapshot_interval", type=int, default=10000, help="Interval of snapshot")
    parser.add_argument("--display_interval", type=int, default=1000, help="Interval of displaying log to console")
    parser.add_argument("--resume", "-r", default="", help="Resume the training from snapshot")
    parser.add_argument("--seed", type=int, default=0, help="Random seed of z at visualization stage")

    args = parser.parse_args()

    print("GPU: {}".format(args.gpu))
    print("# Minibatch-size: {}".format(args.batchsize))
    print("# Iter: {}".format(args.iter))
    print()

    gen = Generator(input_width=256, input_height=256, ch=256)
    dis = Discriminator(input_width=256, input_height=256, ch=512)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    def make_optimizer(model, alpha=0.0002, beta1=0.5, weightdecay=True):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        if weightdecay:
            optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), "hook_dec")
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    train, test = load_train()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    updater = CompGANUpdater(
        tc=29000, td=1000,
        alpha=0.5,
        models=(gen, dis),
        iterator=train_iter,
        optimizer={"gen": opt_gen, "dis": opt_dis},
        device=args.gpu
    )

    trainer = training.Trainer(updater, (args.iter, "iteration"), out=args.out)

    snapshot_interval = (args.snapshot_interval, "iteration")
    display_interval = (args.display_interval, "iteration")

    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)

    trainer.extend(
        extensions.snapshot_object(gen, 'gen_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)

    trainer.extend(
        extensions.snapshot_object(dis, 'dis_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)

    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(
        extensions.PrintReport(['epoch', 'iteration', 'gen/loss', 'dis/loss',]),
        trigger=display_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            gen, test[:16], 4, 4, args.seed, args.out
        ),
        trigger=display_interval
    )

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    initial_image(test[:16], 4, 4, args.seed, args.out)
    trainer.run()

if __name__ == "__main__":
    main()
