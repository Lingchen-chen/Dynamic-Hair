import argparse
import os
from Tools.utils import mkdirs
from Models import get_option_setter, get_option_parser


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        # network arch
        self.parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self.parser.add_argument('--image_size', type=int, default=256, help='input data size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input hair channels per frame')
        self.parser.add_argument('--voxel_depth', type=int, default=96, help='resolution for voxel depth z')
        self.parser.add_argument('--voxel_width', type=int, default=128, help='resolution for voxel width x')
        self.parser.add_argument('--voxel_height', type=int, default=128, help='resolution for voxel height y')
        self.parser.add_argument('--min_channels', type=int, default=16, help='min channels in networks')
        self.parser.add_argument('--max_channels', type=int, default=64, help='max channels in networks')

        self.parser.add_argument('--netG', type=str, default="HairRNNNet", help='the name of the network for geometry regression')
        self.parser.add_argument('--netF', type=str, default="HairWarpNet", help='the name of the network for flow regression')

        # experiment
        self.parser.add_argument('--expr_dir', type=str, default='checkpoints', help='name of the dir to save all the experiments')

        self.initialized = True

    def gather_options(self):
        """Add additional model-specific options"""

        if not self.initialized:
            self.initialize()

        opt, _ = self.parser.parse_known_args()

        # gather other options from different models
        get_option_setter(opt.netG)(self.parser)
        get_option_setter(opt.netF)(self.parser)

        # parse options
        self.opt = self.parser.parse_args()
        get_option_parser(opt.netG)(self.opt)
        get_option_parser(opt.netF)(self.opt)

        self.opt.isTrain = self.isTrain

    def parse(self, save=True):

        self.gather_options()

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if self.opt.voxel_width != self.opt.voxel_height:
            print('invalid voxel resolution. The width and height should be the same')
            exit(0)

        # save to the disk
        expr_dir = self.opt.expr_dir
        mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
