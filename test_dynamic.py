import tensorflow as tf
from DynamicHair import DynamicHair
from TemporalHair import TemporalHair
from Options.test_options import TestOptions


def test():
    opt = TestOptions().parse(save=False)
    opt.batch_size = 1   # test code only supports batchSize = 1
    opt.isTrain = False
    opt.save_model = True
    with tf.Graph().as_default(), tf.Session() as sess:
        if opt.netG == "HairRNNNet":
            Hair = TemporalHair(sess, opt)
        elif opt.netG == "HairSpatNet":
            Hair = DynamicHair(sess, opt)
        else:
            exit(-1)
        Hair.generate_fields(opt.test_video_dir, opt.test_start_frame, opt.test_frames, opt.test_refine_iters)


if __name__ == "__main__":
    test()