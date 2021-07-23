from .base_solver import *
from .networks import HairWarpNet
from Loss.loss import flow_loss, laplacian_smooth3d


class warp_net_data_loader(base_loader):

    def __init__(self, dirs, batch_size, image_size, isTrain=True):
        super().__init__(dirs, batch_size, image_size, isTrain)

        self.prev_gt_flo = None
        self.prev_gt_occ = None
        self.prev_gt_ori = None

        self.normalize = True

        if self.isTrain:
            self.load_thread()

    def _get_one_batch_data(self):

        samples = self.start_generation()  # note the assignment is cite

        gt_flo = []  # N * 128 * 128 * 96
        gt_occ = []  # N * 2 * 128 * 128 * 96
        gt_ori = []  # N * 2 * 128 * 128 * 96

        flip = False
        load = True     # there are some bugs for the data, some frames are invalid
        for d in samples:
            try:
                id, center = d  # str and int
                video_dir = self.videos[id].video_dir
                frames = self.videos[id].frames

                if center == len(frames) - 1:
                    next = center - 1
                elif center == 0:
                    next = center + 1
                else:
                    next = center + (1 if random.random() > 0.5 else -1)

                occ = []
                ori = []

                file_name = os.path.join(video_dir, frames[center])
                if center < next:
                    gt_flo.append(np.expand_dims(get_ground_truth_forward(file_name, flip, normalize=self.normalize), 0))
                else:
                    gt_flo.append(np.expand_dims(get_ground_truth_bacward(file_name, flip, normalize=self.normalize), 0))

                occ.append(np.expand_dims(get_ground_truth_3D_occ(file_name, flip), 0))
                ori.append(np.expand_dims(get_ground_truth_3D_ori(file_name, flip), 0))

                file_name = os.path.join(video_dir, frames[next])
                occ.append(np.expand_dims(get_ground_truth_3D_occ(file_name, flip), 0))
                ori.append(np.expand_dims(get_ground_truth_3D_ori(file_name, flip), 0))

                gt_occ.append(np.expand_dims(np.concatenate(occ, axis=0), 0))
                gt_ori.append(np.expand_dims(np.concatenate(ori, axis=0), 0))

            except:
                load = False
                print("Load Failure!")
                break

        if load:
            gt_flo = np.concatenate(gt_flo, axis=0)
            gt_occ = np.concatenate(gt_occ, axis=0)
            gt_ori = np.concatenate(gt_ori, axis=0)

            self.prev_gt_flo = gt_flo
            self.prev_gt_occ = gt_occ
            self.prev_gt_ori = gt_ori
        else:
            gt_flo = self.prev_gt_flo
            gt_occ = self.prev_gt_occ
            gt_ori = self.prev_gt_ori

        self.end_generation()  # be care of this place

        self.queue.put((gt_flo, gt_occ, gt_ori))

    def get_one_batch_test_data(self, video_dir, evaluation=False):

        frames = get_the_frames(video_dir)

        # yield data
        for i in range(0, len(frames) - 1):

            occs = []
            oris = []
            flow = []

            for ii in range(2):
                file_name = os.path.join(video_dir, frames[i + ii])
                occs.append(get_ground_truth_3D_occ(file_name)[None])
                oris.append(get_ground_truth_3D_ori(file_name)[None])

                if evaluation:
                    if ii == 0:
                        flow.append(get_ground_truth_forward(file_name, False, normalize=self.normalize)[None])
                    else:
                        flow.append(get_ground_truth_bacward(file_name, False, normalize=self.normalize)[None])

            occs = np.concatenate(occs, axis=0)[None]
            oris = np.concatenate(oris, axis=0)[None]

            for ii in range(2):
                if ii == 1:
                    occs = occs[:, ::-1]
                    oris = oris[:, ::-1]
                if not evaluation:
                    yield occs, oris
                else:
                    yield occs, oris, flow[ii]


class HairWarpNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        """Add new options"""

        parser.add_argument('--pyr_lvls', type=int, default=4, help='# of feature pyramid')
        parser.add_argument('--pre_lvls', type=int, default=1, help='the layer for flow prediction: 1 <= <= pyr_lvls')
        parser.add_argument('--flow_reg_weight', type=float, default=1e-5, help='the regularization weight')
        parser.add_argument('--end_to_end', action='store_true', help='train the model in a end2end manner')

        # to do set default end-end

    def initialize(self, sess, opt, name="HairWarpNet"):
        super(HairWarpNetSolver, self).initialize(sess, opt, name)

        self.pyr_lvls = opt.pyr_lvls

        if self.isTrain:
            self.end_to_end = opt.end_to_end
            self.flow_reg_weight = opt.flow_reg_weight

            self.pre_lvls = opt.pre_lvls

            if self.end_to_end:
                self.pre_lvls = 1   # override pre_lvls to 1
            else:
                assert 1 <= self.pre_lvls <= self.pyr_lvls

            self.train_data_loader = warp_net_data_loader(self.train_data_dir, self.batch_size, self.image_size, self.isTrain)
            self.val_data_loader = warp_net_data_loader(self.val_data_dir, self.batch_size, self.image_size, self.isTrain)
        else:
            self.pre_lvls = 1
            self.batch_size = 2 # hard code
            self.test_data_loader = warp_net_data_loader(None, self.batch_size, self.image_size, self.isTrain)

        self.nn = HairWarpNet(pyr_lvls=self.pyr_lvls, pre_lvls=self.pre_lvls, scope_name=self.name,
                              min_channels=self.min_channels, max_channels=self.max_channels)
        self.build_graph()

    def build_graph(self):

        with tf.variable_scope(self.name):
            self.in_occs = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 2, self.depth, self.height, self.width, 1])
            self.in_oris = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 2, self.depth, self.height, self.width, 3])
            self.gt_flow = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 3])

            if self.isTrain:
                in_oris = []
                in_oris.append(tf.expand_dims(close_ori(self.in_oris[:, 0], self.in_occs[:, 0], 3, 1), 1))
                in_oris.append(tf.expand_dims(close_ori(self.in_oris[:, 1], self.in_occs[:, 1], 3, 1), 1))
                self.pr_flow = self.nn(tf.concat(in_oris, axis=1))
            else:
                self.pr_flow = self.nn(self.in_oris * self.in_occs)

            if self.isTrain:
                self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
                boundaries = [self.iterations*0.3, self.iterations*0.5, self.iterations*0.7, self.iterations*0.8, self.iterations*0.9]
                boundaries = [int(i) for i in boundaries]
                values = [self.learning_rate / (2 ** boundary) for boundary in range(len(boundaries) + 1)]
                self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values, 'lr_multisteps')

                self.train_vars, self.fixed_vars = self.nn.get_vars(self.end_to_end, outer_scope_name=self.name)

                self.flow_loss = flow_loss(self.in_occs[:, 0], self.gt_flow, self.pr_flow)  # 0 is the center frame in loader
                self.regu_loss = laplacian_smooth3d(self.pr_flow) * self.flow_reg_weight

                self.total_loss = self.flow_loss + self.regu_loss
                self.total_loss = tf.check_numerics(self.total_loss, 'NaN/Inf in total loss')
                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step)  # actually all the parameters

                summaries = []

                summaries.append(tf.summary.scalar("total_loss", self.total_loss))
                summaries.append(tf.summary.scalar("regu_loss", self.regu_loss))
                summaries.append(tf.summary.scalar("flow_loss", self.flow_loss))
                summaries.append(tf.summary.scalar("learning_rate", self.learning_rate))

                self.step_summaries = tf.summary.merge(summaries)
                self.trainWriter = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)
                self.valWriter = tf.summary.FileWriter(self.val_log_dir)

            else:
                saver = tf.train.Saver(var_list=slim.get_variables(self.name))
                self.load_model(saver)

    def get_feed_dict(self, data_list, evaluation=False):
        if self.isTrain or evaluation:
            gt_flow, gt_occs, gt_oris = data_list
            feed_dict = {self.gt_flow: gt_flow, self.in_occs: gt_occs, self.in_oris: gt_oris}
        else:
            occs, oris = data_list
            feed_dict = {self.in_occs: occs, self.in_oris: oris}
        return feed_dict

    def load_model(self, saver):
        if self.isTrain:
            if self.continue_train:
                if not self.load(saver, self.checkpoint_dir):
                    print("training from scratch.")
            else:
                if self.pre_lvls < self.pyr_lvls and not self.end_to_end:
                    saver_dec = tf.train.Saver(self.fixed_vars)
                    for var in self.fixed_vars:
                        print("fixed", var)
                    for var in self.train_vars:
                        print("train", var)

                    # load the checkpoints of previously trained level
                    if not self.load(saver_dec, self.checkpoint_dir):
                        print("Training is invalid.")
                        return
        else:
            if not self.load(saver, self.checkpoint_dir):
                print("testing is invalid.")