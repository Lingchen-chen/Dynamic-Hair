from .base_solver import *
from .networks import HairTempNet
from Loss.loss import flow_loss, laplacian_smooth3d


class temp_net_data_loader(base_loader):
    def __init__(self, dirs, batch_size, window_size, image_size, isTrain=True):
        super().__init__(dirs, batch_size, image_size, isTrain)

        self.window_size = window_size
        assert (self.window_size >= 2)

        self.prev_inputs = None
        self.prev_gt_for = None
        self.prev_gt_bac = None

        self.prev_gt_for_occ = None
        self.prev_gt_bac_occ = None

        self.normalize = True

        # initialize the thread
        if self.isTrain:
            self.load_thread()

    def _get_one_batch_data(self):

        samples = self.start_generation()

        inputs = []         # N * T * 256 * 256 * 3

        gt_for = []         # N * 128 * 128 * 96 * 3
        gt_bac = []         # N * 128 * 128 * 96 * 3

        gt_for_occ = []     # N * 128 * 128 * 96 * 1
        gt_bac_occ = []     # N * 128 * 128 * 96 * 1

        flip = random.random() > 0.5    # random.random() > 0.5
        noise = random.random() > 0.5
        load = True     # there are some bugs for the data, some frames are invalid
        for d in samples:
            try:
                id, center = d  # str and int
                video_dir = self.videos[id].video_dir
                frames = self.videos[id].frames

                if center + self.window_size > len(frames):
                    center = len(frames) - self.window_size

                sta_2d = center
                end_2d = center + self.window_size

                images = []
                for frame in frames[sta_2d: end_2d]:
                    file_name = os.path.join(video_dir, frame)
                    images.append(np.expand_dims(get_conditional_input_data(file_name, flip, noise, self.image_size), 0))

                images = np.concatenate(images, axis=0)
                inputs.append(np.expand_dims(images, 0))

                file_name = os.path.join(video_dir, frames[end_2d - 2])
                gt_for.append(np.expand_dims(get_ground_truth_forward(file_name, flip, normalize=self.normalize), 0))
                gt_for_occ.append(np.expand_dims(get_ground_truth_3D_occ(file_name, flip), 0))

                file_name = os.path.join(video_dir, frames[end_2d - 1])
                gt_bac.append(np.expand_dims(get_ground_truth_bacward(file_name, flip, normalize=self.normalize), 0))
                gt_bac_occ.append(np.expand_dims(get_ground_truth_3D_occ(file_name, flip), 0))

            except:
                load = False
                print("Load Failure!")
                break

        if load:
            inputs = np.concatenate(inputs, axis=0)
            gt_for = np.concatenate(gt_for, axis=0)
            gt_bac = np.concatenate(gt_bac, axis=0)
            gt_for_occ = np.concatenate(gt_for_occ, axis=0)
            gt_bac_occ = np.concatenate(gt_bac_occ, axis=0)

            self.prev_inputs = inputs
            self.prev_gt_for = gt_for
            self.prev_gt_bac = gt_bac
            self.prev_gt_for_occ = gt_for_occ
            self.prev_gt_bac_occ = gt_bac_occ
        else:
            inputs = self.prev_inputs
            gt_for = self.prev_gt_for
            gt_bac = self.prev_gt_bac

            gt_for_occ = self.prev_gt_for_occ
            gt_bac_occ = self.prev_gt_bac_occ

        self.end_generation()  # be care of this place

        self.queue.put((inputs, gt_for, gt_bac, gt_for_occ, gt_bac_occ))

    def get_one_batch_test_data(self, video_dir, evaluation=False, start_frame=0, test_frames=20):

        frames = get_the_frames(video_dir)
        images = []

        # initialization
        for i in range(self.window_size - 1):
            file_name = os.path.join(video_dir, frames[0])
            print(file_name)
            images.append(np.expand_dims(get_conditional_input_data(file_name, False, False, self.image_size), 0))

        # yield data
        for i in range(1, len(frames)):

            file_name = os.path.join(video_dir, frames[i])
            print(file_name)
            images.append(np.expand_dims(get_conditional_input_data(file_name, False, False, self.image_size), 0))
            in_img = np.expand_dims(np.concatenate(images, axis=0), 0)  # 1 * T * 256 * 256 * 3

            if i - 1 >= start_frame:

                if evaluation:
                    file_name = os.path.join(video_dir, frames[i-1])
                    gt_for = get_ground_truth_forward(file_name, flip=False, normalize=self.normalize)[None]
                    gt_for_occ = get_ground_truth_3D_occ(file_name, flip=False)[None]

                    file_name = os.path.join(video_dir, frames[i])
                    gt_bac = get_ground_truth_bacward(file_name, flip=False, normalize=self.normalize)[None]
                    gt_bac_occ = get_ground_truth_3D_occ(file_name, flip=False)[None]

                    yield in_img, gt_for, gt_bac, gt_for_occ, gt_bac_occ

                else:
                    yield in_img

            if i == start_frame + test_frames - 1:
                break

            images.pop(0)


class HairTempNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        """Add new options and rewrite default values for existing options"""

        parser.add_argument('--flow_in_wsz', type=int, default=2, help='# of input frames for HairTempNet')
        parser.add_argument('--flow_reg_weight', type=float, default=1e-5, help='flow regularization loss')

    def initialize(self, sess, opt, name="HairTempNet"):
        super(HairTempNetSolver, self).initialize(sess, opt, name)

        # class specific args
        self.window_size = opt.flow_in_wsz

        if self.isTrain:
            self.train_data_loader = temp_net_data_loader(self.train_data_dir, self.batch_size, self.window_size, self.image_size, self.isTrain)
            self.val_data_loader = temp_net_data_loader(self.val_data_dir, self.batch_size, self.window_size, self.image_size, self.isTrain)
            self.flow_reg_weight = opt.flow_reg_weight  # 1e-5
        else:
            self.test_data_loader = temp_net_data_loader(None, self.batch_size, self.window_size, self.image_size, self.isTrain)

        self.nn = HairTempNet(self.image_size,
                              self.depth, self.height, self.width, scope_name=self.name,
                              min_channels=self.min_channels, max_channels=self.max_channels)
        self.build_graph()

    def build_graph(self):

        with tf.variable_scope(self.name):
            self.in_img = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.window_size, self.image_size, self.image_size, self.input_nc])
            self.gt_for = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 3])
            self.gt_bac = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 3])
            self.gt_for_occ = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 1])
            self.gt_bac_occ = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 1])

            pr_for, pr_bac = self.nn(self.in_img)

            self.pr_for = pr_for
            self.pr_bac = pr_bac
            if isinstance(self.pr_for, (list, tuple)):
                self.pr_for = self.pr_for[-1]
            if isinstance(self.pr_bac, (list, tuple)):
                self.pr_bac = self.pr_bac[-1]

            if self.isTrain:
                self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
                boundaries = [self.iterations*0.3, self.iterations*0.5, self.iterations*0.7, self.iterations*0.8, self.iterations*0.9]
                boundaries = [int(i) for i in boundaries]
                values = [self.learning_rate / (2 ** boundary) for boundary in range(len(boundaries) + 1)]
                self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values, 'lr_multisteps')

                self.flow_loss_for = flow_loss(self.gt_for_occ, self.gt_for, pr_for)   # to do: fix the bug
                self.regu_loss_for = laplacian_smooth3d(self.pr_for) * self.flow_reg_weight

                self.flow_loss_bac = flow_loss(self.gt_bac_occ, self.gt_bac, pr_bac)
                self.regu_loss_bac = laplacian_smooth3d(self.pr_bac) * self.flow_reg_weight

                self.total_loss = self.flow_loss_for + self.regu_loss_for + self.flow_loss_bac + self.regu_loss_bac
                self.total_loss = tf.check_numerics(self.total_loss, 'NaN/Inf in total loss')
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step)

                summaries = []

                summaries.append(tf.summary.scalar("total_loss", self.total_loss))
                summaries.append(tf.summary.scalar("regu_loss_for", self.regu_loss_for))
                summaries.append(tf.summary.scalar("flow_loss_for", self.flow_loss_for))
                summaries.append(tf.summary.scalar("regu_loss_bac", self.regu_loss_bac))
                summaries.append(tf.summary.scalar("flow_loss_bac", self.flow_loss_bac))
                summaries.append(tf.summary.scalar("learning_rate", self.learning_rate))

                self.step_summaries = tf.summary.merge(summaries)
                self.trainWriter = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
                self.valWriter = tf.summary.FileWriter(self.val_log_dir)

            else:
                saver = tf.train.Saver(var_list=slim.get_variables(self.name))
                self.load_model(saver)

    def get_feed_dict(self, data_list, evaluation=False):
        if self.isTrain or evaluation:
            imgs, fors, bacs, fors_occ, bacs_occ = data_list
            feed_dict = {self.in_img: imgs, self.gt_for: fors, self.gt_bac: bacs,
                         self.gt_for_occ: fors_occ, self.gt_bac_occ: bacs_occ}
        else:
            if isinstance(data_list, (list, tuple)):
                imgs = data_list[0]
            else:
                imgs = data_list
            feed_dict = {self.in_img: imgs}
        return feed_dict