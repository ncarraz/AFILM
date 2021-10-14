import os
import argparse
import models
from models.afilm import get_afilm
from models.tfilm import get_tfilm
from utils import load_h5
import tensorflow as tf
from tensorflow import keras


tf.compat.v1.disable_eager_execution()

def make_parser():
    train_parser = argparse.ArgumentParser()

    train_parser.add_argument('--model', default='afilm',
        choices=('afilm', 'tfilm'),
        help='model to train')
    train_parser.add_argument('--train', required=True,
        help='path to h5 archive of training patches')
    train_parser.add_argument('--val', required=True,
        help='path to h5 archive of validation set patches')
    train_parser.add_argument('-e', '--epochs', type=int, default=20,
        help='number of epochs to train')
    train_parser.add_argument('--batch-size', type=int, default=16,
        help='training batch size')
    train_parser.add_argument('--logname', default='tmp-run',
        help='folder where logs will be stored')
    train_parser.add_argument('--layers', default=4, type=int,
        help='number of layers in each of the D and U halves of the network')
    train_parser.add_argument('--alg', default='adam',
        help='optimization algorithm')
    train_parser.add_argument('--lr', default=3e-4, type=float,
        help='learning rate')
    train_parser.add_argument('--save_path', default="model.h5",
        help='path to save the model')
    train_parser.add_argument('--r', type=int, default=4, help='upscaling factor')
    train_parser.add_argument('--pool_size', type=int, default=4, help='size of pooling window')
    train_parser.add_argument('--strides', type=int, default=4, help='pooling stide')
    return train_parser

class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, file_path):
        super(CustomCheckpoint, self).__init__()
        self.file_path = file_path
    
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.file_path)

def train(args):
    X_train, Y_train = load_h5(args.train)
    X_val, Y_val = load_h5(args.val)

    model = get_model(args)
    opt = keras.optimizers.Adam(learning_rate=args.lr)
    model_checkpoint_callback = CustomCheckpoint(file_path=args.save_path)
    model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.epochs, callbacks=[model_checkpoint_callback])


def get_model(args):
    if args.model == 'tfilm':
        model = get_tfilm(n_layers=args.layers, scale=args.r)
    elif args.model == 'afilm':
        model = get_afilm(n_layers=args.layers, scale=args.r)
    else:
        raise ValueError('Invalid model')
    return model


def main():
    parser = make_parser()
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()