from Models.HairRNNNetSolver import HairRNNNetSolver
from Models.ops import warp_voxel
from Models.HairWarpNetSolver import HairWarpNetSolver
from Tools.MarchingCubes import save_hair_model
from Tools.utils import *
import tensorflow as tf


class TemporalHair():

    def __init__(self, sess, opt):

        self.opt = opt
        self.sess = sess
        self.isTrain = opt.isTrain

        self.HairNetSolver = HairRNNNetSolver()
        self.WarpNetSolver = HairWarpNetSolver()

        self.depth = opt.voxel_depth
        self.height = opt.voxel_height
        self.width = opt.voxel_width

        if not self.isTrain:
            self.HairNetSolver.initialize(sess, opt)
            self.WarpNetSolver.initialize(sess, opt)
            self.test_thresh = opt.test_thresh
            self.window_size = opt.build_tc_wsz
            self.padding = self.window_size // 2
            self.save_model = opt.save_model
            self.build_warping_block()

    def train_hair_geom_net(self):

        if self.isTrain:
            self.HairNetSolver.initialize(self.sess, self.opt)
            self.HairNetSolver.train()

    def train_hair_flow_net(self):
        if self.isTrain:
            self.WarpNetSolver.initialize(self.sess, self.opt)
            self.WarpNetSolver.train()

    def generate_fields(self, video_dir, start_frame=0, test_frames=100, refine_iterations=2):

        loader = self.HairNetSolver.test_data_loader.get_one_batch_test_data(video_dir, start_frame=start_frame)
        occs = []
        oris = []

        pr_ori = next(loader)
        for frameI, images in enumerate(loader):
            print(images.shape)
            feed_dict = self.HairNetSolver.get_feed_dict(images, pr_ori)
            occ, ori, ori_f = self.sess.run([self.HairNetSolver.out_occ,
                                             self.HairNetSolver.out_ori_o,
                                             self.HairNetSolver.out_ori], feed_dict=feed_dict)
            pr_ori = np.concatenate([pr_ori[:, 1:], ori_f[:, np.newaxis]], axis=1)
            occs.append(occ)
            oris.append(ori)
            if frameI == test_frames // 2:
                save_hair_model(occ.squeeze(), "SSS.obj", 0.3, False)

            if frameI >= test_frames - 1:
                break

        forwards = None
        bacwards = None
        occs = np.concatenate(occs, axis=0)
        oris = np.concatenate(oris, axis=0)
        for i in range(refine_iterations):
            forwards = []
            bacwards = []
            for frameI in range(test_frames-1):
                in_occs = occs[frameI:frameI+2][None] > 0.3
                in_oris = oris[frameI:frameI+2][None]
                in_occs = np.concatenate([in_occs, in_occs[:, ::-1]], axis=0)
                in_oris = np.concatenate([in_oris, in_oris[:, ::-1]], axis=0)

                feed_dict = {self.WarpNetSolver.in_occs: in_occs, self.WarpNetSolver.in_oris: in_oris}
                pr_flow = self.sess.run(self.WarpNetSolver.pr_flow, feed_dict=feed_dict)

                forwards.append(pr_flow[:1])
                bacwards.append(pr_flow[1:])

            forwards = forwards + [np.zeros_like(forwards[-1])]
            bacwards = [np.zeros_like(bacwards[-1])] + bacwards

            forwards = np.concatenate(forwards, axis=0)
            bacwards = np.concatenate(bacwards, axis=0)

            self.build_temporal_coherece(occs, forwards, bacwards)

        for frameI in range(test_frames):

            occ = (occs[frameI] > self.test_thresh).astype(np.float32)
            ori = oris[frameI] * occ
            forward = forwards[frameI] * occ / np.array([1., -1., -1.])
            bacward = bacwards[frameI] * occ / np.array([1., -1., -1.])

            if self.WarpNetSolver.test_data_loader.normalize:
                forward /= stepInv
                bacward /= stepInv

            save_path = os.path.join(video_dir, f"frame{frameI + start_frame}")

            if self.save_model:
                save_hair_model(occ.squeeze(), os.path.join(save_path, "HairMesh.obj"), self.test_thresh, False)

            scipy.io.savemat(os.path.join(save_path, "Occ3D.mat"),
                             {"Occ": occ.squeeze().transpose(1, 2, 0).astype(np.float64)},
                             do_compression=True)
            scipy.io.savemat(os.path.join(save_path, "Ori3D.mat"),
                             {"Ori": ori.transpose(1, 2, 3, 0).reshape(128, 128, 288).astype(np.float64)},
                             do_compression=True)
            scipy.io.savemat(os.path.join(save_path, "ForwardWarp.mat"),
                             {"Warp": forward.transpose(1, 2, 3, 0).reshape(128, 128, 288).astype(np.float64)},
                             do_compression=True)
            scipy.io.savemat(os.path.join(save_path, "BacwardWarp.mat"),
                             {"Warp": bacward.transpose(1, 2, 3, 0).reshape(128, 128, 288).astype(np.float64)},
                             do_compression=True)

    def build_temporal_coherece(self, occs, forwards, bacwards):

        for i in range(self.padding, occs.shape[0]-self.padding):

            feed_dict = {self.target: occs[i:i+1], self.occs_f: occs[i-self.padding:i], self.fflows: forwards[i-self.padding:i],
                         self.occs_b: occs[i+self.padding:i:-1], self.bflows: bacwards[i+self.padding:i:-1]}
            occs[i] = self.sess.run(self.grads, feed_dict=feed_dict)

            if i == occs.shape[0] // 2:
                save_hair_model(occs[i].squeeze(), "SSSS.obj", 0.3, False)

    def build_warping_block(self):
        # key idea is that
        # Frame_(t-3) --> Frame_(t)
        # first consecutively forwards warp the warping fields from Frame_(t-2) to Frame_(t-1):
        #       new warp field for frame t-3 : F_(t-3) = W(F_(t-2), F_(t-3)) + F_(t-3)
        #                                      F_(t-3) = W(F_(t-1), F_(t-3)) + F_(t-3)
        # second warp the occ in frame t to frame t-3: Occ_hat = W(Occ_(t), F_(t-3))
        # then calculate the gradients of Occ_hat with respect to Occ_(t)

        self.target = tf.placeholder(dtype=tf.float32, shape=[1, self.depth, self.height, self.width, 1])
        self.occs_f = tf.placeholder(dtype=tf.float32, shape=[self.padding, self.depth, self.height, self.width, 1])
        self.occs_b = tf.placeholder(dtype=tf.float32, shape=[self.padding, self.depth, self.height, self.width, 1])
        self.fflows = tf.placeholder(dtype=tf.float32, shape=[self.padding, self.depth, self.height, self.width, 3])
        self.bflows = tf.placeholder(dtype=tf.float32, shape=[self.padding, self.depth, self.height, self.width, 3])

        def get_warped_flow(flows):
            cond = lambda t, f, fs: t < tf.shape(fs)[0]
            body = lambda t, f, fs: [t+1, f+warp_voxel(fs[t:t+1], f), fs]
            return tf.while_loop(cond, body, [tf.constant(1), flows[:1], flows], back_prop=False)[1]

        occss = tf.concat([self.occs_f, self.occs_b], axis=0)
        flows = tf.concat([get_warped_flow(self.fflows[t:]) for t in range(self.padding)] +
                          [get_warped_flow(self.bflows[t:]) for t in range(self.padding)], axis=0)
        targs = tf.concat([self.target] * self.padding * 2, axis=0)
        warps = warp_voxel(targs, flows)
        grads = tf.gradients(warps, targs, occss)[0]
        grads = tf.concat([grads, self.target], axis=0)

        self.grads = tf.reduce_mean(grads, axis=0)
