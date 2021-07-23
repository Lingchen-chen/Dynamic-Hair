from Models.ops import *
from queue import Queue
from threading import Thread
from Tools.utils import *


class base_loader:

    def __init__(self, dirs, batch_size, image_size=256, isTrain=True):

        self.batch_size = batch_size
        self.image_size = image_size
        self.isTrain = isTrain

        if self.isTrain:
            self.videos = get_all_the_videos(dirs)
            self.videos_num = len(self.videos)

            self.train_corpus = []
            self.train_nums = 0

            self.generate_corpus()
            self.current_pos = 0

            self.queue = Queue()
            self.thread = None

    def generate_corpus(self):

        # exclude the tail, to do improve this
        for id, video in enumerate(self.videos):
            for pos in range(len(video.frames)):
                self.train_corpus.append((id, pos))  # str int
        random.shuffle(self.train_corpus)
        self.train_nums = len(self.train_corpus)
        print(f"num of training data: {self.train_nums}")

    def start_generation(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.current_pos + batch_size > self.train_nums:
            self.current_pos = self.train_nums - batch_size
        return self.train_corpus[self.current_pos:self.current_pos + batch_size]

    def end_generation(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        self.current_pos += batch_size
        if self.current_pos >= self.train_nums:
            self.current_pos = 0
            random.shuffle(self.train_corpus)

    def load_thread(self):
        self.thread = Thread(target=self._get_one_batch_data)
        self.thread.start()

    def get_one_batch_data(self, load_new=True):

        if self.thread:
            self.thread.join()  # Main thread should wait for sub thread finishes, after which the sub thread dies

        if load_new:
            self.load_thread()

        return self.queue.get()

    def _get_one_batch_data(self):
        pass


class BaseSolver:

    @staticmethod
    def modify_options(parser):
        pass

    @staticmethod
    def parse_opt(opt):
        pass

    def initialize(self, sess, opt, name="base"):

        self.sess = sess    # tensorflow session
        self.name = name

        self.batch_size = opt.batch_size
        self.depth = opt.voxel_depth
        self.width = opt.voxel_width
        self.height = opt.voxel_height
        self.input_nc = opt.input_nc
        self.image_size = opt.image_size
        self.min_channels = opt.min_channels
        self.max_channels = opt.max_channels

        self.root_dir = os.path.join(opt.expr_dir, self.name)
        self.train_log_dir = os.path.join(self.root_dir, 'logs', 'train')
        self.val_log_dir = os.path.join(self.root_dir, 'logs', 'val')
        self.checkpoint_dir = os.path.join(self.root_dir, 'checkpoint')

        mkdirs([self.root_dir, self.train_log_dir, self.val_log_dir, self.checkpoint_dir])

        self.isTrain = opt.isTrain
        if self.isTrain:
            self.save_iter = opt.save_latest_freq
            self.iterations = opt.niter
            self.display_iter = opt.display_freq
            self.learning_rate = opt.lr
            self.continue_train = opt.continue_train

            # the train data dir & vai data dir are shared for each sub class
            self.train_data_dir = opt.train_data_dir
            self.val_data_dir = opt.val_data_dir

    def load(self, saver, checkpoint_dir):
        print(" [*] Reading latest checkpoint from folder %s." % (checkpoint_dir))
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print("Load Failure!")
            return False

    def convert_to_mask(self, occ):
        occ = tf.reshape(occ, [self.batch_size * self.depth * self.height * self.width * 1])
        positive = tf.sigmoid(occ) > 0.5
        resutImg = tf.where(positive, tf.ones_like(occ), tf.zeros_like(occ))
        resutImg = tf.reshape(resutImg, [self.batch_size, self.depth, self.height, self.width, 1])

        return resutImg

    def get_occ_slice(self, x, sliceId=40):
        sliceImg = tf.slice(x, [0, sliceId, 0, 0, 0], [self.batch_size, 1, self.height, self.width, 1])
        sliceImg = tf.reshape(sliceImg, [self.batch_size * self.height * self.width])
        positive = tf.sigmoid(sliceImg) > 0.5
        resutImg = tf.where(positive, tf.ones_like(sliceImg), tf.zeros_like(sliceImg))
        resutImg = tf.reshape(resutImg, [self.batch_size, self.height, self.width, 1])

        return resutImg

    def get_ori_slice(self, x, occ=None, sliceId=40):
        sliceImg = tf.slice(x, [0, sliceId, 0, 0, 0], [self.batch_size, 1, self.height, self.width, 3])
        sliceImg = tf.squeeze((sliceImg + 1.0) * 0.5)

        if occ is not None:
            maskOImg = tf.slice(occ, [0, sliceId, 0, 0, 0], [self.batch_size, 1, self.height, self.width, 1])
            maskOImg = tf.reshape(maskOImg, [self.batch_size * self.height * self.width])
            positive = tf.sigmoid(maskOImg) > 0.5
            resutImg = tf.where(positive, tf.ones_like(maskOImg), tf.zeros_like(maskOImg))
            resutImg = tf.reshape(resutImg, [self.batch_size, self.height, self.width, 1])
            sliceImg = resutImg * sliceImg

        return tf.clip_by_value(sliceImg, 0, 1)

    def get_feed_dict(self, data_list):
        pass

    def load_model(self, saver):

        if self.isTrain:
            if self.continue_train:
                if not self.load(saver, self.checkpoint_dir):
                    print(" Training from Scratch! ")
        else:
            if not self.load(saver, self.checkpoint_dir):
                print("testing is invalid.")

    def train(self):

        with tf.name_scope(name=self.name):

            # Initialize
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            save_path = os.path.join(self.checkpoint_dir, 'model.ckpt')

            self.load_model(saver)

            for iteration in range(self.sess.run(self.global_step) + 1, self.iterations + 1):

                feed_dict = self.get_feed_dict(self.train_data_loader.get_one_batch_data())
                if hasattr(self, 'use_gan'):
                    if self.use_gan:
                        self.sess.run(self.train_step_D, feed_dict=feed_dict)
                self.sess.run(self.train_step, feed_dict=feed_dict)

                if iteration % self.display_iter == 0:
                    feed_dict = self.get_feed_dict(self.train_data_loader.get_one_batch_data())
                    summaries = self.sess.run(self.step_summaries, feed_dict=feed_dict)
                    self.trainWriter.add_summary(summaries, global_step=iteration)

                    feed_dict = self.get_feed_dict(self.val_data_loader.get_one_batch_data())
                    summaries = self.sess.run(self.step_summaries, feed_dict=feed_dict)
                    self.valWriter.add_summary(summaries, global_step=iteration)

                if iteration % self.save_iter == 0:
                    saver.save(self.sess, save_path, global_step=iteration)

            saver.save(self.sess, save_path, global_step=self.iterations)