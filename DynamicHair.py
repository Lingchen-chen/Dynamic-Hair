from Models import find_model_using_name
from Models.ops import warp_voxel
from Tools.utils import *
from Tools.MarchingCubes import save_hair_model
import tensorflow as tf


class DynamicHair():

    def __init__(self, sess, opt):

        self.opt = opt
        self.sess = sess
        self.isTrain = opt.isTrain

        self.HairGeomNet = find_model_using_name(opt.netG)()
        self.HairFlowNet = find_model_using_name(opt.netF)()

        self.depth = opt.voxel_depth
        self.height = opt.voxel_height
        self.width = opt.voxel_width

        if not self.isTrain:
            self.HairGeomNet.initialize(sess, opt)
            self.HairFlowNet.initialize(sess, opt)
            self.test_thresh = opt.test_thresh
            self.window_size = opt.build_tc_wsz
            self.padding = self.window_size // 2
            self.save_model = opt.save_model
            self.build_warping_block()

    def train_hair_geom_net(self):

        if self.isTrain:
            self.HairGeomNet.initialize(self.sess, self.opt)
            self.HairGeomNet.train()

    def train_hair_flow_net(self):
        if self.isTrain:
            self.HairFlowNet.initialize(self.sess, self.opt)
            self.HairFlowNet.train()

    def generate_fields(self, video_dir, start_frame=0, test_frames=100, refine_iterations=1):

        loader = self.HairGeomNet.test_data_loader.get_one_batch_test_data(video_dir, start_frame=start_frame, test_frames=test_frames)

        # first generate occ & ori
        occs = []
        oris = []
        frameNames = []
        for frameI, data in enumerate(loader):
            images, frameName = data
            print(images.shape)
            feed_dict = self.HairGeomNet.get_feed_dict(images)
            occ, ori = self.sess.run([self.HairGeomNet.out_occ,
                                      self.HairGeomNet.out_ori], feed_dict=feed_dict)

            occs.append(occ)
            oris.append(ori)
            frameNames.append(frameName)

        occs = np.concatenate(occs, axis=0)
        oris = np.concatenate(oris, axis=0)

        # second generate flow
        forwards = []
        bacwards = []
        if self.opt.netF == "HairTempNet":
            loader = self.HairFlowNet.test_data_loader.get_one_batch_test_data(video_dir, start_frame=start_frame, test_frames=test_frames)
            for frameI, images in enumerate(loader):
                feed_dict = self.HairFlowNet.get_feed_dict(images)
                pr_for, pr_bac = self.sess.run([self.HairFlowNet.pr_for,
                                                self.HairFlowNet.pr_bac], feed_dict=feed_dict)

                forwards.append(pr_for)
                bacwards.append(pr_bac)

            forwards = forwards + [np.zeros_like(forwards[-1])]
            bacwards = [np.zeros_like(bacwards[-1])] + bacwards

            forwards = np.concatenate(forwards, axis=0)
            bacwards = np.concatenate(bacwards, axis=0)

        for i in range(refine_iterations):

            if self.opt.netF == "HairWarpNet":
                forwards = []
                bacwards = []

                for frameI in range(test_frames-1):
                    in_occs = occs[frameI:frameI+2][None] > self.test_thresh
                    in_oris = oris[frameI:frameI+2][None]
                    in_occs = np.concatenate([in_occs, in_occs[:, ::-1]], axis=0)
                    in_oris = np.concatenate([in_oris, in_oris[:, ::-1]], axis=0)

                    feed_dict = self.HairFlowNet.get_feed_dict([in_occs, in_oris])
                    pr_flow = self.sess.run(self.HairFlowNet.pr_flow, feed_dict=feed_dict)

                    forwards.append(pr_flow[:1])
                    bacwards.append(pr_flow[1:])

                forwards = forwards + [np.zeros_like(forwards[-1])]
                bacwards = [np.zeros_like(bacwards[-1])] + bacwards

                forwards = np.concatenate(forwards, axis=0)
                bacwards = np.concatenate(bacwards, axis=0)

            occs = self.build_temporal_coherece(occs, forwards, bacwards)

        for frameI in range(test_frames):

            occ = (occs[frameI] > self.test_thresh).astype(np.float32)
            ori = oris[frameI] * occ
            forward = forwards[frameI] * occ / np.array([1., -1., -1.])
            bacward = bacwards[frameI] * occ / np.array([1., -1., -1.])

            if self.HairFlowNet.test_data_loader.normalize:
                forward /= stepInv
                bacward /= stepInv

            save_path = os.path.join(video_dir, frameNames[frameI])

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

        return occs

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
