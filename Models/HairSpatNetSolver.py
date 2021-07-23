from .base_solver import *
from .networks import HairSpatNet, Discriminator
from Loss.loss import uniform_sample_loss, laplacian_smooth3d, adversarial_loss, binary_cross_entropy
from Loss.LovaszSoftmax import lovasz_hinge


class spat_net_data_loader(base_loader):
    def __init__(self, dirs, batch_size, window_size, image_size, isTrain=True):
        super().__init__(dirs, batch_size, image_size, isTrain)

        self.window_size = window_size
        assert (self.window_size % 2)
        self.pad = self.window_size // 2

        self.prev_inputs = None
        self.prev_gt_occ = None
        self.prev_gt_ori = None

        # initialize the thread
        if self.isTrain:
            self.load_thread()

    def _get_one_batch_data(self):

        samples = self.start_generation()  # note the assignment is cite

        inputs = []  # N * T * 256 * 256 * 3
        gt_occ = []  # N * 128 * 128 * 96
        gt_ori = []  # N * 128 * 128 * 96 * 3

        flip = random.random() > 0.5    # random.random() > 0.5
        noise = random.random() > 0.5
        load = True     # there are some bugs for the data, some frames are invalid
        for d in samples:
            try:
                id, center = d  # str and int
                video_dir = self.videos[id].video_dir
                frames = self.videos[id].frames

                sta_2d = center - self.pad
                end_2d = center + 1 + self.pad

                low_2d = max(sta_2d, 0)
                hig_2d = min(end_2d, len(frames))

                pad_lef_2d = low_2d - sta_2d
                pad_rig_2d = end_2d - hig_2d

                images = []
                for frame in frames[low_2d: hig_2d]:
                    file_name = os.path.join(video_dir, frame)
                    # print("2D:", file_name)
                    images.append(np.expand_dims(get_conditional_input_data(file_name, flip, noise, self.image_size), 0))

                images = np.concatenate(images, axis=0)
                images = np.pad(images, ((pad_lef_2d, pad_rig_2d), (0, 0), (0, 0), (0, 0)), 'edge')
                inputs.append(np.expand_dims(images, 0))

                file_name = os.path.join(video_dir, frames[center])
                print("3D:", file_name)
                gt_occ.append(np.expand_dims(get_ground_truth_3D_occ(file_name, flip), 0))
                gt_ori.append(np.expand_dims(get_ground_truth_3D_ori(file_name, flip), 0))

            except:
                load = False
                print("Load Failure!")
                break

        if load:
            inputs = np.concatenate(inputs, axis=0)
            gt_occ = np.concatenate(gt_occ, axis=0)
            gt_ori = np.concatenate(gt_ori, axis=0)

            self.prev_inputs = inputs
            self.prev_gt_occ = gt_occ
            self.prev_gt_ori = gt_ori
        else:
            inputs = self.prev_inputs
            gt_occ = self.prev_gt_occ
            gt_ori = self.prev_gt_ori

        self.end_generation()
        self.queue.put((inputs, gt_occ, gt_ori))

    def get_one_batch_test_data(self, video_dir, evaluation=False, start_frame=0, test_frames=20):

        frames = get_the_frames(video_dir)
        images = []

        # initialization
        # -2, -1, 0, 1, 2 (pad = 2, wsz = 5)
        for i in range(self.window_size):
            file_name = os.path.join(video_dir, frames[0] if i < self.pad else frames[i - self.pad])
            print(file_name)
            images.append(np.expand_dims(get_conditional_input_data(file_name, False, False, self.image_size), 0))

        # yield data
        for i in range(len(frames)):

            inputs = np.expand_dims(np.concatenate(images, axis=0), 0)  # T * 256 * 256 * 3

            if i >= start_frame:
                if not evaluation:
                    yield inputs, frames[i]

                else:
                    file_name = os.path.join(video_dir, frames[i])
                    gt_occ = np.expand_dims(get_ground_truth_3D_occ(file_name), 0)
                    gt_ori = np.expand_dims(get_ground_truth_3D_ori(file_name), 0)

                    yield inputs, gt_occ, gt_ori, frames[i]

            if i == start_frame + test_frames - 1:
                break

            images.pop(0)

            idx = i + self.pad + 1  # add new image
            if idx >= len(frames):
                idx = len(frames) - 1

            file_name = os.path.join(video_dir, frames[idx])
            print(file_name)
            images.append(np.expand_dims(get_conditional_input_data(file_name, False, False, self.image_size), 0))


class HairSpatNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        """Add new class-specific options"""
        parser.add_argument('--temporal_widths', type=str, default='1', help='# of input frames, e.g. 3,3,3. means 27 frames, -1 means no temporal info')
        parser.add_argument('--cls_weight', type=float, default=1.0, help='classification loss weight')
        parser.add_argument('--ori_weight', type=float, default=1.0, help='orientation regression loss')
        parser.add_argument('--ori_smooth_weight', type=float, default=1e-5, help='orientation regularization loss')
        parser.add_argument('--ori_content_weight', type=float, default=0.01, help='orientation content loss')
        parser.add_argument('--use_gan', action='store_true', help='whether use gan or not for the ori branch')

    @staticmethod
    def parse_opt(opt):
        """class-specific parse function"""
        temporal_widths = []

        for t in [int(n) for n in opt.temporal_widths.split(',')]:
            if t % 2 == 1 and t > 1:
                temporal_widths.append(t)

        opt.temporal_widths = temporal_widths

    def initialize(self, sess, opt, name="HairSpatNet"):
        super(HairSpatNetSolver, self).initialize(sess, opt, name)  # this name is very important for managing the variables, etc

        self.temporal_widths = opt.temporal_widths

        # calculate the window size for each batch data
        if len(self.temporal_widths) == 0:
            self.window_size = 1
        else:
            self.window_size = reduce(lambda x, y: x*y, [1] + self.temporal_widths)

        if self.isTrain:
            self.cls_weight = opt.cls_weight
            self.ori_weight = opt.ori_weight
            self.ori_smooth_weight = opt.ori_smooth_weight
            self.ori_content_weight = opt.ori_content_weight

            self.train_data_loader = spat_net_data_loader(self.train_data_dir, self.batch_size, self.window_size, self.image_size, self.isTrain)
            self.val_data_loader = spat_net_data_loader(self.val_data_dir, self.batch_size, self.window_size, self.image_size, self.isTrain)

            self.use_gan = opt.use_gan
            if self.use_gan:
                self.name_D = "Discriminator"
                self.nn_D = Discriminator(self.image_size, self.depth, self.height, self.width, scope_name=self.name_D)
        else:
            self.test_data_loader = spat_net_data_loader(None, self.batch_size, self.window_size, self.image_size, self.isTrain)

        self.name_G = "Generator"
        self.nn = HairSpatNet(self.image_size,
                              self.depth, self.height, self.width, self.temporal_widths, self.name_G,
                              self.min_channels, self.max_channels)
        self.build_graph()

    def build_graph(self):
        with tf.variable_scope(self.name):
            self.in_img = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.window_size, self.image_size, self.image_size, self.input_nc])
            self.gt_occ_raw = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 1])
            self.gt_ori_raw = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 3])
            self.gt_occ = close(self.gt_occ_raw, 3)
            self.gt_ori = close_ori(self.gt_ori_raw, self.gt_occ_raw, 3)

            self.out_occ, self.out_ori_o = self.nn(self.in_img)     # N * D * H * W * 1, note this occ is not logit

            if self.isTrain:
                self.out_ori = self.out_ori_o * self.gt_occ
            else:
                self.out_occ = tf.nn.sigmoid(10 * self.out_occ)    # enlarge the range in order to avoid border effect when warped
                self.out_ori = self.out_ori_o

            if self.isTrain:
                self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
                boundaries = [self.iterations*0.3, self.iterations*0.5, self.iterations*0.7, self.iterations*0.8, self.iterations*0.9]
                boundaries = [int(i) for i in boundaries]
                values = [self.learning_rate / (2 ** boundary) for boundary in range(len(boundaries) + 1)]
                self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values, 'lr_multisteps')

                self.cls_loss = lovasz_hinge(self.out_occ, self.gt_occ) * self.cls_weight
                self.ori_loss = uniform_sample_loss(self.gt_occ, self.gt_ori, self.out_ori) * self.ori_weight
                self.ori_smooth_loss = laplacian_smooth3d(self.out_ori_o) * self.ori_smooth_weight

                self.G_loss = self.G_FM = self.D_loss = self.GP = 0
                if self.use_gan:
                    self.G_loss, self.G_FM, self.D_loss, self.GP = adversarial_loss(self.nn_D, self.gt_ori, self.out_ori, self.in_img, self.gt_occ)

                self.total_loss = self.cls_loss + self.ori_loss + self.ori_smooth_loss + self.G_loss + self.G_FM * self.ori_content_weight
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step,
                                                                                      var_list=slim.get_variables(f"{self.name}/{self.name_G}"))

                if self.use_gan:
                    self.total_loss_D = self.D_loss + self.GP
                    self.train_step_D = tf.train.AdamOptimizer(self.learning_rate*3).minimize(self.total_loss_D,
                                                                                              var_list=slim.get_variables(f"{self.name}/{self.name_D}"))
                # ------------------------------ log define ---------------------------- #
                loss_summary = []

                loss_summary.append(tf.summary.scalar("cls_loss", self.cls_loss))
                loss_summary.append(tf.summary.scalar("ori_loss", self.ori_loss))
                loss_summary.append(tf.summary.scalar("ori_smooth_loss", self.ori_smooth_loss))
                loss_summary.append(tf.summary.scalar("G_loss", self.G_loss))
                loss_summary.append(tf.summary.scalar("G_FM", self.G_FM))
                loss_summary.append(tf.summary.scalar("D_loss", self.D_loss))
                loss_summary.append(tf.summary.scalar("GP", self.GP))
                loss_summary.append(tf.summary.scalar("learning_rate", self.learning_rate))

                image_summary = []
                sliceId = tf.random_uniform([], self.depth // 4, self.depth // 4 * 3, dtype=tf.int32)
                image_summary.append(tf.summary.image("in_imgs", self.in_img[:, self.window_size//2]))
                image_summary.append(tf.summary.image("out_occ_slice", self.get_occ_slice(self.out_occ, sliceId)))
                image_summary.append(tf.summary.image("gt_occ_slice", self.get_occ_slice(self.gt_occ, sliceId)))
                image_summary.append(tf.summary.image("out_ori_slice", self.get_ori_slice(self.out_ori_o, self.out_occ, sliceId)))
                image_summary.append(tf.summary.image("gt_ori_slice", self.get_ori_slice(self.gt_ori, self.gt_occ, sliceId)))

                self.step_summaries = tf.summary.merge(loss_summary + image_summary)
                self.trainWriter = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
                self.valWriter = tf.summary.FileWriter(self.val_log_dir)

            else:
                saver = tf.train.Saver(var_list=slim.get_variables(self.name))
                self.load_model(saver)

    def get_feed_dict(self, data_list, evaluation=False):
        if self.isTrain or evaluation:
            images, occupancy, orientation = data_list
            feed_dict = {self.in_img: images, self.gt_occ_raw: occupancy, self.gt_ori_raw: orientation}
        else:
            if isinstance(data_list, (list, tuple)):
                imgs = data_list[0]
            else:
                imgs = data_list
            feed_dict = {self.in_img: imgs}

        return feed_dict