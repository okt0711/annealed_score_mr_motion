import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--act_type', type=str, default='lrelu', help='type of activation function')
        self.parser.add_argument('--ngf', type=int, default=64, help='the number of generator filters of the first layer')
        self.parser.add_argument('--norm_type', type=str, default='instance', help='normalization method (batch, instance, none)')
        self.parser.add_argument('--save_path', type=str, default='cycle/ckpt', help='the path for saving results')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt
