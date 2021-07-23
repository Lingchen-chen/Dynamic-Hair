from Models.base_solver import *
from Models.networks import HairRNNNet
from Loss.loss import uniform_sample_loss, laplacian_smooth3d
from Loss.LovaszSoftmax import lovasz_hinge


class rnn_net_data_loader(base_loader):
    def __init__(self, dirs, batch_size, input_size, image_size, intervals=None, isTrain=True):
        super().__init__(dirs, batch_size, image_size, isTrain)

        self.input_size = input_size
        self.prev_size = input_size - 1

        if self.isTrain:
            # generate a video list, each video consists of many tiny videos
            # whose length is fixed and based on current hierachy
            self.intervals = intervals.keys()
            self.hierarchical_videos = []
            for num, interval in intervals.items():
                videos = get_all_the_videos(dirs, interval=interval+self.prev_size)
                random.shuffle(videos)
                self.hierarchical_videos.append(videos)

            print(self.intervals)

            self.prev_inputs = None
            self.prev_gt_occ = None
            self.prev_gt_ori = None

            self.current_hierarchy = 0
            self.current_video_pos = 0
            self.current_video_sze = None
            self.current_videos = None

    def calculate_current_hierarchy(self, iters):
        self.current_hierarchy = 0
        for interval in self.intervals:
            if iters // interval == 0:
                break
            self.current_hierarchy += 1

        print("current_hierarchy:", self.current_hierarchy)

    def start_video_generation(self, iters):

        self.calculate_current_hierarchy(iters)
        current_total_sze = len(self.hierarchical_videos[self.current_hierarchy])
        current_total_videos = self.hierarchical_videos[self.current_hierarchy]

        if self.current_video_pos + self.batch_size >= current_total_sze:
            self.current_video_pos = current_total_sze - self.batch_size

        self.current_videos = current_total_videos[self.current_video_pos: self.current_video_pos + self.batch_size]
        self.current_video_sze = len(self.current_videos[0].frames) - self.prev_size
        for i in range(1, self.batch_size):
            self.current_video_sze = min(self.current_video_sze, len(self.current_videos[i].frames) - self.prev_size)

        self.current_pos = self.prev_size

        self.prev_inputs = None
        self.prev_gt_occ = None
        self.prev_gt_ori = None

        self.thread = None
        self.load_thread()

        print("current_video_size", self.current_video_sze)

        return self.init_video_params()

    def init_video_params(self):

        pr_ori = []

        for video in self.current_videos:   # a batch
            video_dir = video.video_dir
            frames = video.frames

            oris = []
            for i in range(self.prev_size):
                file_name = os.path.join(video_dir, frames[i])
                print("init", file_name)
                oris.append(get_ground_truth_3D_ori(file_name, flip=False)[None])

            pr_ori.append(np.concatenate(oris, axis=0)[None])

        pr_ori = np.concatenate(pr_ori, axis=0)

        return pr_ori, self.current_video_sze

    def end_video_generation(self):

        current_total_sze = len(self.hierarchical_videos[self.current_hierarchy])
        current_total_videos = self.hierarchical_videos[self.current_hierarchy]

        self.current_video_pos += self.batch_size
        if self.current_video_pos >= current_total_sze:
            self.current_video_pos = 0
            random.shuffle(current_total_videos)

        # at the end of generation, directly kill the current thread
        if self.thread:
            self.thread.join()  # block

        # since there might be still data in the queue, directly ignore them
        while not self.queue.empty():
            self.queue.get()

    def _get_one_batch_data(self):

        inputs = []  # N * 256 * 256 * T*3
        gt_occ = []  # N * 96 * 128 * 128 * 1
        gt_ori = []  # N * 96 * 128 * 128 * 3

        flip = False  # random.random() > 0.5
        load = True
        for video in self.current_videos:
            try:
                video_dir = video.video_dir
                frames = video.frames

                sta = self.current_pos - self.prev_size
                end = self.current_pos + 1

                occs = []
                oris = []
                images = []
                for i in range(sta, end):
                    file_name = os.path.join(video_dir, frames[i])
                    print("2D", file_name)
                    images.append(get_conditional_input_data(file_name, flip=flip, random_noise=True)[None])
                    if i == end - 1:
                        print("3D", file_name)
                        occs.append(get_ground_truth_3D_occ(file_name, flip)[None])
                        oris.append(get_ground_truth_3D_ori(file_name, flip)[None])

                images = np.concatenate(images, axis=0)
                occs = np.concatenate(occs, axis=0)
                oris = np.concatenate(oris, axis=0)

                gt_occ.append(np.expand_dims(occs, 0))
                gt_ori.append(np.expand_dims(oris, 0))
                inputs.append(np.expand_dims(images, 0))

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

        self.current_pos += 1
        self.queue.put((inputs, gt_occ, gt_ori))

    def get_one_batch_test_data(self, video_dir, evaluation=False, start_frame=0):

        frames = get_the_frames(video_dir)

        # initialization
        images = []
        pr_ori = []
        if start_frame < self.prev_size:
            file_name = os.path.join(video_dir, frames[start_frame])
            images = [get_conditional_input_data(file_name, flip=False, random_noise=False)[None]] * self.prev_size
            pr_ori = [get_ground_truth_3D_ori(file_name, flip=False)[None]] * self.prev_size
        else:
            for i in range(start_frame - self.prev_size, start_frame):
                file_name = os.path.join(video_dir, frames[i])
                images.append(get_conditional_input_data(file_name, flip=False, random_noise=False)[None])
                pr_ori.append(get_ground_truth_3D_ori(file_name, flip=False)[None])

        pr_ori = np.concatenate(pr_ori, axis=0)[None]
        yield pr_ori

        # yield data
        for i in range(start_frame, len(frames)):
            file_name = os.path.join(video_dir, frames[i])
            images.append(get_conditional_input_data(file_name, flip=False, random_noise=False)[None])
            inputs = np.concatenate(images, axis=0)[None]  # 1 * T * 256 * 256 * 3

            if not evaluation:
                yield inputs
            else:
                gt_occ = np.expand_dims(get_ground_truth_3D_occ(file_name), 0)
                gt_ori = np.expand_dims(get_ground_truth_3D_ori(file_name), 0)

                yield inputs, gt_occ, gt_ori

            images.pop(0)


class HairRNNNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--input_size', type=int, default=2, help='# of input frames for geometry')
        parser.add_argument('--cls_weight', type=float, default=1.0, help='classification loss weight')
        parser.add_argument('--ori_weight', type=float, default=1.0, help='orientation regression loss')
        parser.add_argument('--ori_smooth_weight', type=float, default=1e-5, help='orientation regularization loss')

        parser.add_argument('--n_step', type=int, default=20000, help='# of every iters to lengthen the sequence')
        parser.add_argument('--n_frames_max', type=int, default=48, help='# of max frames')

    def initialize(self, sess, opt, name="HairRNN"):
        super().initialize(sess, opt, name)

        self.input_size = opt.input_size
        self.prev_size = opt.input_size - 1

        if self.isTrain:
            self.cls_weight = opt.cls_weight
            self.ori_weight = opt.ori_weight
            self.ori_smooth_weight = opt.ori_smooth_weight

            self.interval = opt.n_step
            self.n_frames_max = opt.n_frames_max
            intervals = {}
            for i in range(self.iterations//self.interval):
                intervals[(i + 1) * self.interval] = min(self.n_frames_max, 2**i * 3)
            self.intervals = intervals
            self.train_data_loader = rnn_net_data_loader(self.train_data_dir, self.batch_size, self.input_size, self.image_size,
                                                         intervals=self.intervals, isTrain=self.isTrain)
        else:
            self.test_data_loader = rnn_net_data_loader(None, self.batch_size, self.input_size, self.image_size, None, False)

        self.nn = HairRNNNet(self.input_size, self.prev_size,
                             self.input_nc, self.image_size,
                             self.depth, self.height, self.width, scope_name=self.name)
        self.build_graph()

    def build_graph(self):
        with tf.name_scope(self.name):
            self.in_img = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.image_size, self.image_size, self.input_size*self.input_nc])
            self.pr_ori = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, self.prev_size*3])
            self.gt_occ = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 1])
            self.gt_ori = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 3])

            self.out_occ_o, self.out_ori_o = self.nn(self.in_img, self.pr_ori)  # N * D * H * W * 1, note this occ is not logit

            self.out_occ = tf.nn.sigmoid(10. * self.out_occ_o)
            self.out_ori = self.out_ori_o * self.convert_to_mask(10. * self.out_occ_o)

            if self.isTrain:
                self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
                boundaries = [self.iterations*0.3, self.iterations*0.5, self.iterations*0.7, self.iterations*0.8, self.iterations*0.9]
                boundaries = [int(i) for i in boundaries]
                values = [self.learning_rate / (2 ** boundary) for boundary in range(len(boundaries) + 1)]
                self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values, 'lr_multisteps')

                self.cls_loss = lovasz_hinge(self.out_occ_o, close(self.gt_occ, 3)) * self.cls_weight
                self.ori_loss = uniform_sample_loss(self.gt_occ, self.gt_ori, self.out_ori_o * self.gt_occ) * self.ori_weight
                self.ori_smooth_loss = laplacian_smooth3d(self.out_ori_o) * self.ori_smooth_weight
                self.total_loss = self.cls_loss + self.ori_loss + self.ori_smooth_loss
                self.total_loss = tf.check_numerics(self.total_loss, 'NaN/Inf in total loss')

                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step)

                # ------------------------------ log define ---------------------------- #
                loss_summary = []

                loss_summary.append(tf.summary.scalar("cls_loss", self.cls_loss))
                loss_summary.append(tf.summary.scalar("ori_loss", self.ori_loss))
                loss_summary.append(tf.summary.scalar("ori_smooth_loss", self.ori_smooth_loss))
                loss_summary.append(tf.summary.scalar("total_loss", self.total_loss))

                image_summary = []
                sliceId = tf.random_uniform([], 30, 60, dtype=tf.int32)
                image_summary.append(tf.summary.image("out_occ_slice", self.get_occ_slice(self.out_occ_o, sliceId)))
                image_summary.append(tf.summary.image("gt_occ_slice", self.get_occ_slice(self.gt_occ, sliceId)))
                image_summary.append(tf.summary.image("out_ori_slice", self.get_ori_slice(self.out_ori_o, self.out_occ_o, sliceId)))
                image_summary.append(tf.summary.image("gt_ori_slice", self.get_ori_slice(self.gt_ori, self.gt_occ, sliceId)))

                self.step_summaries = tf.summary.merge(loss_summary + image_summary)
                self.trainWriter = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)

            else:
                saver = tf.train.Saver(var_list=slim.get_variables(self.name))
                self.load(saver, self.checkpoint_dir)

    def train(self):

        with tf.name_scope(name=self.name):

            # ------------------------------ Initialize ---------------------------- #
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())
            save_path = os.path.join(self.checkpoint_dir, 'model.ckpt')

            if not self.load(saver, self.checkpoint_dir):
                print(" Training from Scratch! ")

            iteration = self.sess.run(self.global_step) + 1
            # ------------------------------ Training ---------------------------- #
            while iteration <= self.iterations:

                # init pr_ori
                pr_ori, video_size = self.train_data_loader.start_video_generation(iteration)
                for frameI in range(video_size):
                    # load data, when the video ends, we do not need new thread to get new data
                    inputs, gt_occ, gt_ori = self.train_data_loader.get_one_batch_data(frameI < video_size - 1)

                    # feed forward running
                    feed_dict = self.get_feed_dict(inputs, pr_ori, gt_occ, gt_ori)
                    _, out_ori = self.sess.run([self.train_step, self.out_ori], feed_dict=feed_dict)

                    # update pr_ori
                    pr_ori = np.concatenate([pr_ori[:, 1:], out_ori[:, np.newaxis]], axis=1)

                    if iteration % self.display_iter == 0:
                        summaries = self.sess.run(self.step_summaries, feed_dict=feed_dict)
                        self.trainWriter.add_summary(summaries, global_step=iteration)

                    if iteration % self.save_iter == 0:
                        saver.save(self.sess, save_path, global_step=iteration)

                    iteration += 1

                self.train_data_loader.end_video_generation()

            saver.save(self.sess, save_path, global_step=self.iterations)

    def get_feed_dict(self, inputs, pr_ori, gt_occ=None, gt_ori=None):

        feed_dict = {}
        feed_dict[self.in_img] = inputs.transpose(0, 2, 3, 1, 4).reshape(*self.in_img.get_shape().as_list())
        feed_dict[self.pr_ori] = pr_ori.transpose(0, 2, 3, 4, 1, 5).reshape(*self.pr_ori.get_shape().as_list())

        if self.isTrain:
            feed_dict[self.gt_occ] = gt_occ.transpose(0, 2, 3, 4, 1, 5).reshape(*self.gt_occ.get_shape().as_list())
            feed_dict[self.gt_ori] = gt_ori.transpose(0, 2, 3, 4, 1, 5).reshape(*self.gt_ori.get_shape().as_list())

        return feed_dict
