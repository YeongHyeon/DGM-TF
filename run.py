import os, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='0'
warnings.filterwarnings('ignore')

import source.datamanager as dman
import source.neuralnet as nn
import source.tf_process as tfp

def main():

    dataset = dman.Dataset(normalize=FLAGS.datnorm)
    neuralnet = nn.DGM(height=dataset.height, width=dataset.width, channel=dataset.channel, \
        ksize=FLAGS.ksize, zdim=FLAGS.zdim, learning_rate=FLAGS.lr, path='Checkpoint')

    neuralnet.confirm_params(verbose=False)
    # neuralnet.confirm_bn()

    tfp.training(neuralnet=neuralnet, dataset=dataset, \
        epochs=FLAGS.epoch, batch_size=FLAGS.batch, normalize=True)
    tfp.test(neuralnet=neuralnet, dataset=dataset, \
        batch_size=FLAGS.batch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--ksize', type=int, default=3, help='Size of Kernel')
    parser.add_argument('--zdim', type=int, default=2, help='Dimension of latent vector z')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--batch', type=int, default=32, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    main()
