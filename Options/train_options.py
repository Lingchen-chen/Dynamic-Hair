from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # data_loader
        self.parser.add_argument('--train_data_dir', type=str, default='D:/HairData/Train')
        self.parser.add_argument('--val_data_dir', type=str, default='D:/HairData/Val')

        # learning rate and loss weight
        self.parser.add_argument('--niter', type=int, default=150000, help='training iterations')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')

        # display the results
        self.parser.add_argument('--display_freq', type=int, default=500, help='frequency of showing training results on screen')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')

        # train which network, the networks are trained separatly currently, to do: train together
        self.parser.add_argument('--train_flow', action='store_true')
        self.parser.add_argument('--continue_train', action='store_true')

        self.parser.set_defaults(train_flow=False)
        self.parser.set_defaults(continue_train=False)

        self.isTrain = True