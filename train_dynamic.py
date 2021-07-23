import tensorflow as tf
from DynamicHair import DynamicHair
from TemporalHair import TemporalHair
from Options.train_options import TrainOptions


def train():
    opt = TrainOptions().parse(save=True)
    with tf.Graph().as_default(), tf.Session() as sess:
        if opt.netG == "HairRNNNet":
            Hair = TemporalHair(sess, opt)
        elif opt.netG == "HairSpatNet":
            Hair = DynamicHair(sess, opt)
        else:
            exit(-1)

        if opt.train_flow:
            Hair.train_hair_flow_net()
        else:
            Hair.train_hair_geom_net()


if __name__ == "__main__":
    train()