from Models.base_solver import *
import cv2


class GrowingNet:

    def __init__(self, voxel_size=[96, 128, 128], local_size=32, stride=16, min_cha=16, max_cha=256, sample_mode="Tri"):

        self.voxel_size = np.array(voxel_size, dtype=np.int32)
        self.local_size = local_size
        self.stride = stride
        self.sample_mode = sample_mode

        assert self.stride % 2 == 0
        assert self.local_size % 2 == 0
        # assert self.local_size / self.stride >= 2                   # for 32 // 16, just overlapping
        assert self.voxel_size[0] % self.local_size == 0            # currently only work for even number
        assert self.voxel_size[1] % self.local_size == 0
        assert self.voxel_size[2] % self.local_size == 0

        self.latent_size = self.voxel_size // self.stride + 1       # since we pad the input
        print("latent_size", self.latent_size)

        self.n_layers = np.log2(self.local_size).astype(np.int32)   # to get the bottleneck representation
        self.min_cha = min_cha
        self.max_cha = max_cha
        print("num of layers", self.n_layers, "min_cha", self.min_cha, "max_cha", self.max_cha)

    def get_ori_slices(self, ori):
        """
        return local oris, each of which has the shape of local_size
        :param ori has the shape of B * D * H * W * 3
        """
        with tf.name_scope("Slicing"):
            centers = []
            latents = []
            B, D, H, W, C = ori.get_shape().as_list()
            d, h, w = [self.local_size] * 3
            ori = tf.pad(ori, ((0, 0), (d//2, d//2), (h//2, h//2), (w//2, w//2), (0, 0)))

            for z in range(self.latent_size[0]):
                for y in range(self.latent_size[1]):
                    for x in range(self.latent_size[2]):
                        beg = [z*self.stride, y*self.stride, x*self.stride]
                        end = [beg[i]+self.local_size for i in range(3)]
                        latents.append(tf.expand_dims(tf.strided_slice(ori, [0]+beg+[0], [B]+end+[3]), 1))
                        centers.append(tf.convert_to_tensor(np.array(beg), dtype=tf.float32))

            centers = tf.concat(centers, axis=0)
            centers = tf.reshape(centers, (1, *self.latent_size, 3))
            centers = tf.tile(centers, (B, 1, 1, 1, 1))
            centers = tf.reverse(centers, axis=[-1])  # to become compatible with strands points x y z coordinate

            latents = tf.concat(latents, axis=1)
            latents = tf.reshape(latents, (B, *self.latent_size, d, h, w, 3))

            return centers, latents

    def encoder(self, ori):
        cha = self.min_cha
        # first get local oris and corresponding global centers
        centers, local_oris = self.get_ori_slices(ori)
        # second encode the local oris into latents
        with tf.variable_scope("Encoder", reuse=tf.AUTO_REUSE):
            latents = tf.reshape(local_oris, (-1, self.local_size, self.local_size, self.local_size, 3))
            with slim.arg_scope([slim.conv3d], activation_fn=leaky_relu):
                for _ in range(self.n_layers - 1):
                    latents = slim.conv3d(latents, cha, 3, 2)
                    cha = min(2*cha, self.max_cha)
                ker = latents.get_shape().as_list()[1:4]
                latents = slim.conv3d(latents, cha, ker, padding='VALID', activation_fn=None)

        latents = tf.reshape(latents, (-1, *self.latent_size, cha))
        # third return
        return centers, latents

    def decoder(self, s, step, wcenters, wlatents):
        cha = self.max_cha
        with tf.variable_scope("Decoder", reuse=tf.AUTO_REUSE):
            r = 2. / tf.to_float(self.local_size)
            p = r * (s - wcenters)                              # [-1, 1],  see paper
            t = step / tf.to_float(self.local_size)             # [0, 1],   see paper
            p = tf.concat([p, t], axis=-1)
            x = tf.concat([wlatents, p], axis=-1)
            with slim.arg_scope([slim.conv2d], activation_fn=leaky_relu):
                for i in range(3):
                    x = slim.conv2d(x, cha, 1)
                    x = tf.concat([x, p], axis=-1)
                    cha //= 2
                x = slim.conv2d(x, 3, 1, activation_fn=None)

        return x / r + wcenters       # note the change of coordinate sys

    def warp_feature(self, s, step, centers, latents):
        """
        warp feature from latents, which has the shape of B * latent_size * C
        :param s: B * N * P * 3
        :param latents: latent features B * latent_size * C
        """
        with tf.name_scope("Warp"):

            def my_sample(NoInputHere, zz, yy, xx):
                wcenters = get_voxel_value(centers, zz, yy, xx)
                wlatents = get_voxel_value(latents, zz, yy, xx)
                return self.decoder(s, step, wcenters, wlatents)

            # since the first center is 0, 0, 0
            ss = s / tf.to_float(self.stride)           # be care that the coordinate of p is x, y, z
            return linear_sample(None, ss, my_sample, self.sample_mode,
                                 D=self.latent_size[0], H=self.latent_size[1], W=self.latent_size[2])

    def nn(self, strands, steps, ori, reuse=None):

        with tf.variable_scope("GrowingNet", reuse=reuse):
            # first cut ori into slices, with or without overlapping
            centers, latents = self.encoder(ori)
            points = self.warp_feature(strands, steps, centers, latents)
            return points

    def rnn(self, starting_points, steps, ori, reuse=None):
        steps = tf.split(steps, steps.get_shape()[-2].value, axis=-2)  # keep the dim
        try:
            starting_points.get_shape()[:-1].merge_with(steps[0].get_shape()[:-1])
        except ValueError:
            raise ValueError("Shape incompatible")

        strands = []
        points = starting_points
        with tf.variable_scope("GrowingNet", reuse=reuse):
            # first cut ori into slices, with or without overlapping
            centers, latents = self.encoder(ori)
            for step in steps:
                points = self.warp_feature(points, step, centers, latents)
                strands.append(points)
        strands = tf.concat(strands, axis=-2)
        return strands


class growing_net_data_loader(base_loader):

    def __init__(self, dirs, pt_per_strand, sd_per_batch, batch_size, isTrain):
        super().__init__(dirs, batch_size, isTrain=isTrain)

        self.pt_num = pt_per_strand
        self.sd_num = sd_per_batch

        if self.isTrain:
            self.load_thread()

    def _get_one_batch_data(self):

        samples = self.start_generation()  # note the assignment is cite

        gt_hairUVMask = []  # N * 128 * 128
        gt_hairUVImage = []  # N * 128 * 128 * 24 * 3
        gt_orientation = []  # N * 128 * 128 * 96 * 1
        gt_distanceField = []  # N * 128 * 128 * 96 * 1

        for d in samples:
            id, center = d  # str and int
            video_dir = self.videos[id].video_dir
            frames = self.videos[id].frames

            file_name = os.path.join(video_dir, frames[center])
            print("3D:", file_name)
            mask, image = get_ground_truth_hair_image(file_name, self.pt_num)
            gt_hairUVMask.append(mask[None])
            gt_hairUVImage.append(image[None])
            gt_orientation.append(get_ground_truth_3D_ori(file_name, False)[None])
            gt_distanceField.append(get_ground_truth_3D_dist(file_name, False)[None])

        gt_hairUVMask = np.concatenate(gt_hairUVMask, axis=0)
        gt_hairUVImage = np.concatenate(gt_hairUVImage, axis=0)
        gt_orientation = np.concatenate(gt_orientation, axis=0)
        gt_distanceField = np.concatenate(gt_distanceField, axis=0)

        self.end_generation()

        if self.sd_num is None:
            self.queue.put((gt_hairUVMask, gt_hairUVImage, gt_orientation, gt_distanceField))
        else:
            gt_strands = []
            for i in range(self.batch_size):
                strands = gt_hairUVImage[[i], gt_hairUVMask[i] > 0]
                np.random.shuffle(strands)  # return a None
                gt_strands.append(strands[:self.sd_num][None])
            gt_strands = np.concatenate(gt_strands, axis=0)

            self.queue.put((gt_strands, gt_orientation, gt_distanceField))


class GrowingNetSolver(BaseSolver):

    @staticmethod
    def modify_options(parser):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--local_size', type=int, default=32, help='res for the local voxel')
        parser.add_argument('--stride', type=int, default=16, help='stride between adjacent local voxels')
        parser.add_argument('--pt_per_strand', type=int, default=24, help='# of points per strand')
        parser.add_argument('--sd_per_batch', type=int, default=1000, help='# of sampled strands per batch')

        parser.add_argument('--n_step', type=int, default=10000, help='# of every iters to lengthen the sequence')
        parser.add_argument('--n_frames_max', type=int, default=24, help='# of max frames')

    def initialize(self, sess, opt, name="GrowingNet"):
        super().__init__(sess, opt, name)

        self.local_size = opt.local_size
        self.stride = opt.stride

        self.pt_num = opt.pt_per_strand
        self.sd_num = opt.sd_per_batch

        self.train_data_loader = growing_net_data_loader(self.train_data_dir, self.pt_num, self.sd_num, self.batch_size, self.isTrain)

        self.GrowingNet = GrowingNet([self.depth, self.height, self.width], self.local_size, self.stride)
        self.build_graph()

    def build_graph(self):

        with tf.name_scope(name="GrowingNet"):
            self.in_oris = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 3])
            self.in_dist = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.depth, self.height, self.width, 1])
            self.in_strands = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.sd_num, self.pt_num, 3])
            self.sta_points = self.in_strands[:, :, :+1]
            self.pre_points = self.in_strands[:, :, :-1]
            self.aft_points = self.in_strands[:, :, +1:]
            self.point_step = tf.linalg.norm(self.aft_points - self.pre_points, axis=-1, keepdims=True)

            self.out_points_1 = self.GrowingNet.nn(self.pre_points, self.point_step, self.in_oris)
            self.out_points_2 = self.GrowingNet.rnn(self.sta_points, self.point_step, self.in_oris, reuse=True)

            if self.isTrain:
                self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
                boundaries = [self.iterations*0.3, self.iterations*0.5, self.iterations*0.7, self.iterations*0.8, self.iterations*0.9]
                boundaries = [int(i) for i in boundaries]
                print(boundaries)
                values = [self.learning_rate / (2 ** boundary) for boundary in range(len(boundaries) + 1)]
                self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values, 'lr_multisteps')

                self.p_loss = tf.reduce_mean(tf.abs(self.aft_points - self.out_points_1))
                self.d_loss = tf.reduce_mean(linear_sample(self.in_dist, self.out_points_2))
                self.total_loss = tf.check_numerics(self.p_loss + self.d_loss, 'NaN/Inf in total loss')

                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step)

                self.step_summaries = tf.summary.merge([tf.summary.scalar("p_loss", self.p_loss),
                                                        tf.summary.scalar("d_loss", self.d_loss)])

                self.train_writer = tf.summary.FileWriter(self.train_log_dir, self.sess.graph)

    def train(self):

        # ------------------------------ Initialize ---------------------------- #
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        save_path = os.path.join(self.checkpoint_dir, 'model.ckpt')

        if not self.load(saver, self.checkpoint_dir):
            print(" Training from Scratch! ")

        # ------------------------------ Training ---------------------------- #
        for iteration in range(self.sess.run(self.global_step), self.iterations):

            strands, oris, dist = self.train_data_loader.get_one_batch_data()
            feed_dict = {self.in_strands: strands, self.in_oris: oris, self.in_dist: dist}
            self.sess.run(self.train_step, feed_dict=feed_dict)
            print(iteration)

            if iteration % 20 == 0:
                strands, oris, dist = self.train_data_loader.get_one_batch_data()
                feed_dict = {self.in_strands: strands, self.in_oris: oris, self.in_dist: dist}
                gts, ots_1, ots_2, summaries = self.sess.run([self.aft_points, self.out_points_1, self.out_points_2, self.step_summaries], feed_dict=feed_dict)
                self.draw_samples(gts[0], ots_1[0], "sss")
                self.draw_samples(gts[1], ots_2[1], "rrr")
                self.train_writer.add_summary(summaries, global_step=iteration)

            if iteration % self.save_iter == 0 and iteration != 0:
                saver.save(self.sess, save_path, global_step=iteration)

        saver.save(self.sess, save_path, global_step=self.iterations)

    def draw_samples(self, strands_in, strands_ot, name="sss"):

        N = strands_in.shape[0]
        s = np.random.randint(0, N)

        s_in = strands_in[s]
        s_ot = strands_ot[s]

        bbmin = np.array([0, 0, 0], dtype=np.float32)
        bbmax = np.array([self.width-1, self.height-1, self.depth-1], dtype=np.float32)

        s_in = np.maximum(np.minimum(np.round(s_in), bbmax), bbmin).astype(np.int32)
        s_ot = np.maximum(np.minimum(np.round(s_ot), bbmax), bbmin).astype(np.int32)

        img1 = np.zeros(shape=[self.height, self.width], dtype=np.uint8)
        img2 = np.zeros(shape=[self.depth, self.height], dtype=np.uint8)

        img1[s_in[:, 1], s_in[:, 0]] = 120
        img1[s_ot[:, 1], s_ot[:, 0]] = 250

        img2[s_in[:, 2], s_in[:, 1]] = 120
        img2[s_ot[:, 2], s_ot[:, 1]] = 250

        cv2.imwrite(f"{name}_1.jpg", img1)
        cv2.imwrite(f"{name}_2.jpg", img2)
