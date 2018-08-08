import os
import argparse
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np

from chainer import cuda
from chainer.training import Trainer
from chainer.training import extensions

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.generator, self.discriminator = kwargs.pop('models')
        super(Updater, self).__init__(*args, **kwargs)

    def generator_loss(self, generator, y_gened):
        batch_size = len(y_gened)
        loss = F.sum(F.softplus(-y_gened)) / batch_size
        chainer.report({'loss': loss}, generator)
        return loss

    def discriminator_loss(self, discriminator, y_gened, y_real):
        batch_size = len(y_gened)
        loss = (F.sum(F.softplus(-y_real)) / batch_size) + (F.sum(F.softplus(y_gened)) / batch_size)
        chainer.report({'loss': loss}, discriminator)
        return loss

    def update(self):
        gen_optimizer = self.get_optimizer('generator')
        dis_optimizer = self.get_optimizer('discriminator')

        next_b = self.get_iterator('main').next()
        batch_size = len(next_b)
        x_real = chainer.Variable(self.converter(next_b, self.device)) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)
        generator, discriminator = self.generator, self.discriminator
        z = chainer.Variable(xp.asarray(generator.random_hidden(batch_size)))
        x_fake = generator(z)
        y_real = discriminator(x_real)
        y_gened = discriminator(x_fake)

        dis_optimizer.update(self.discriminator_loss, discriminator, y_gened, y_real)
        gen_optimizer.update(self.generator_loss, generator, y_gened)

class Generator(chainer.Chain):
    def __init__(self, hidden, width=3, ch=512, scale=0.02):
        super(Generator, self).__init__()
        w = chainer.initializers.Normal(scale)
        self.hidden = hidden
        self.ch = ch
        self.width = width
 
        with self.init_scope():
            self.fc0 = L.Linear(hidden, width*width*ch, w)

            self.de1 = L.Deconvolution2D(ch, ch//2, 2, 2, 1, w)
            self.bn1 = L.BatchNormalization(ch)

            self.de2 = L.Deconvolution2D(ch//2, ch//4, 2, 2, 1, w)
            self.bn2 = L.BatchNormalization(ch//2)

            self.de3 = L.Deconvolution2D(ch//4, ch//8, 2, 2, 1, w)
            self.bn3 = L.BatchNormalization(ch//4)

            self.de4 = L.Deconvolution2D(ch//8, 1, 3, 3, 1, w) 
            self.bn4 = L.BatchNormalization(ch//8)
 
    def __call__(self, z):
        h = self.fc0(z)
        h = F.reshape(h, (len(z), self.ch, self.width, self.width))
        h = F.relu(self.bn1(h))
        h = F.relu(self.bn2(self.de1(h)))
        h = F.relu(self.bn3(self.de2(h)))
        h = F.relu(self.bn4(self.de3(h)))
        h = F.sigmoid(self.de4(h))
        return h

    def random_hidden(self, batch_size):
        return np.random.uniform(-1, 1, (batch_size, self.hidden, 1, 1)).astype(np.float32)

class Discriminator(chainer.Chain):
    def __init__(self, width=3, ch=512, scale=0.02):
        super(Discriminator, self).__init__()
        w = chainer.initializers.Normal(scale)

        with self.init_scope():
            self.cn0 = L.Convolution2D(1, 64, 3, 3, 1, w)

            self.cn1 = L.Convolution2D(ch//8, 128, 2, 2, 1, w)
            self.bn1 = L.BatchNormalization(ch//4, False)

            self.cn2 = L.Convolution2D(ch//4, 256, 2, 2, 1, w)
            self.bn2 = L.BatchNormalization(ch//2, False)

            self.cn3 = L.Convolution2D(ch//2, 512, 2, 2, 1, w)
            self.bn3 = L.BatchNormalization(ch//1, False)

            self.fc4 = L.Linear(None, 1, w)
 
    def __call__(self, x):
        h = F.leaky_relu(self.cn0(x))
        h = F.leaky_relu(self.bn1(self.cn1(h)))
        h = F.leaky_relu(self.bn2(self.cn2(h)))
        h = F.leaky_relu(self.bn3(self.cn3(h)))
        h = self.fc4(h)
        return h

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--hidden', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
 
    generator = Generator(hidden=args.hidden)
    discriminator = Discriminator()
    chainer.cuda.get_device_from_id(args.gpu).use()
    generator.to_gpu() 
    discriminator.to_gpu()
 
    gen_optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    gen_optimizer.setup(generator)
    gen_optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
    dis_optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    dis_optimizer.setup(discriminator)
    dis_optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
 
    train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=3, scale=255.)
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    updater = Updater(models=(generator, discriminator), iterator=train_iter, optimizer={'generator':gen_optimizer, 'discriminator':dis_optimizer}, device=args.gpu)
    trainer = Trainer(updater, (args.epoch, 'epoch'), out="result")

    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'generator/loss', 'discriminator/loss',]), trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=(1, 'epoch')))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        generator, 'gen_epoch_{.updater.epoch}.npz'), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        discriminator, 'dis_epoch_{.updater.epoch}.npz'), trigger=(1, 'epoch'))

    trainer.run()
 
if __name__ == '__main__':
    main()