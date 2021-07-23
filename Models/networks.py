from Models.ops import *
import numpy as np


class HairSpatNet(UNet):
    def __init__(self,
                 image_size=256,    # input information
                 depth=96, height=128, width=128,   # voxel information
                 temporal_widths=None,              # temporal information
                 scope_name="HairSpatNet",          # outer scope name
                 min_channels=16,
                 max_channels=64):

        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')

        super(HairSpatNet, self).__init__(**layer_kwargs)

    def __call__(self, x, reuse=None):

        with tf.variable_scope(self.scope_name):
            if self.temporalModel:
                caches = self.encoder(x, reuse)
            else:
                caches = self.encoder_regular(x, reuse)

            occ = self.decoder2(caches.copy(), 1, reuse=reuse, scope="Decoder_For_Occ")
            ori = self.decoder2(caches.copy(), 3, reuse=reuse, scope="Decoder_For_Ori")
            ori = pixel_norm(ori)
            occ = occ * 10.
            return occ, ori


class HairTempNet(UNet):
    def __init__(self,
                 image_size=256,    # input information
                 depth=96, height=128, width=128,   # voxel information
                 temporal_widths=None,              # temporal information
                 scope_name="HairTempNet",          # outer scope name
                 min_channels=16,
                 max_channels=64):

        layer_kwargs = locals()
        layer_kwargs.pop('self')
        layer_kwargs.pop('__class__')

        super(HairTempNet, self).__init__(**layer_kwargs)

    def __call__(self, x, reuse=None):

        with tf.variable_scope(self.scope_name):
            if x.shape.ndims == 5:
                shape = x.get_shape()[1:]
                if not shape.is_fully_defined():
                    raise ValueError('Inputs %s has undefined last dimensions %s.' % (x.name, shape))
                T, H, W, C = shape.as_list()

                x = tf.transpose(x, [0, 2, 3, 1, 4])
                x = tf.reshape(x, [-1, H, W, T*C])

            caches = self.encoder_regular(x, reuse)
            forward = self.decoder(caches.copy(), 3, reuse=reuse, scope="Forward")
            bacward = self.decoder(caches.copy(), 3, reuse=reuse, scope="Bacward")

            return forward, bacward


class HairWarpNet(object):

    def __init__(self, **kwargs):

        self.pyr_lvls = kwargs.pop("pyr_lvls", 4)
        self.pre_lvls = kwargs.pop("pre_lvls", 1)
        self.scope_name = kwargs.pop("scope_name", "HairWarpNet")

        self.min_channels = kwargs.pop("min_channels", 16)
        self.max_channels = kwargs.pop("max_channels", 64)

    def __call__(self, voxels):
        return self.nn(voxels)

    def extract_features(self, voxel, scope='FeaturePyramid'):
        assert (1 <= self.pyr_lvls <= 4)

        x = voxel
        pyr_feats = [None]
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv3d],
                                activation_fn=leaky_relu):
                for lvl in range(1, self.pyr_lvls + 1):
                    chan = min(self.max_channels, self.min_channels * (2**(lvl-1)))
                    x = slim.conv3d(x, chan, 4, 2, scope=f'conv{lvl}_2')
                    pyr_feats.append(x)

        return pyr_feats

    def warp(self, c2, flow, scope='Warp'):
        """
        Compute the warped feature from c2.
        :param c2: B * D * H * W * C    the source feature
        :param flow: B * D * H * W * 3  the flow from c1 to c2
        :return: B * D * H * W * C      warped feature
        """
        with tf.name_scope(scope):
            return warp_voxel(c2, flow)

    def deconv(self, x, lvl, scope='UpFlow'):
        """
        Up sample the flow.
        :param x: B * D * H * W * 3
        :return:  B * D * 2H * 2W * 3
        """
        scope = f'{scope}{lvl}'
        with tf.name_scope(scope):
            return sample(x) * 2.

    def predict_flow_refining(self, c1, c2, m_flow, lvl, scope="Refining"):
        """
        Estimate optical flow, the second step.
        :param c1: B * D * H * W * C        The feature from Voxel1 at level lvl
        :param c2: B * D * H * W * C        The feature from Voxel2 at level lvl
        :param m_flow: B * D * H * W * 3    predicted flow from matching step
        """
        scope = f'{scope}{lvl}'
        with tf.variable_scope(scope):
            chan = c2.get_shape()[-1]

            # first warp c2
            if m_flow is None:
                m_flow = 0.
                x = tf.concat([c1, c2], axis=-1)
            else:
                c2 = self.warp(c2, m_flow)
                x = tf.concat([c1, c2, m_flow], axis=-1)

            # second compute flow
            with slim.arg_scope([slim.conv3d],
                                activation_fn=leaky_relu):
                x = slim.conv3d(x, chan//1, 3, 1, scope=f'conv{lvl}_1')
                x = slim.conv3d(x, chan//2, 3, 1, scope=f'conv{lvl}_2')
                x = slim.conv3d(x, chan//4, 3, 1, scope=f'conv{lvl}_3')

                delta = slim.conv3d(x, 3, 3, 1, activation_fn=None, scope=f'conv{lvl}_4',
                                    weights_initializer=tf.zeros_initializer(),
                                    biases_initializer=None)

            return delta + m_flow

    def nn(self, voxels, reuse=None):
        """
        Predict the warp field
        :param voxels: B * 2 * D * H * W * 3
        :return:
        """
        with tf.variable_scope(self.scope_name, reuse=reuse):

            c1 = self.extract_features(voxels[:, 0])
            c2 = self.extract_features(voxels[:, 1])

            with tf.variable_scope("FlowPrediction"):
                u_flow = None
                # f_pyrs = []
                for lvl in range(self.pyr_lvls, self.pre_lvls - 1, -1):
                    # cost matching needs a lot of memory, remove it for simplicity
                    # m_flow = self.predict_flow_matching(c1[lvl], c2[lvl], u_flow, lvl)
                    # r_flow = self.predict_flow_refining(c1[lvl], c2[lvl], m_flow, lvl)

                    r_flow = self.predict_flow_refining(c1[lvl], c2[lvl], u_flow, lvl)
                    u_flow = self.deconv(r_flow, lvl)

                return u_flow

    def get_vars(self, end_2_end=False, outer_scope_name="HairWarpNet"):

        E_vars = slim.get_variables(f"{outer_scope_name}/{self.scope_name}/FeaturePyramid")
        F_vars = slim.get_variables(f"{outer_scope_name}/{self.scope_name}/FlowPrediction")

        if end_2_end or self.pre_lvls == self.pyr_lvls:
            return E_vars + F_vars, []
        else:
            # only vars of current level need training
            tvars = []
            fvars = E_vars
            for v in F_vars:
                if v.name.find(f"Refining{self.pre_lvls}") != -1:
                    tvars.append(v)
                elif v.name.find(f"Matching{self.pre_lvls}") != -1:
                    tvars.append(v)
                elif v.name.find(f"UpFlow{self.pre_lvls}") != -1:
                    tvars.append(v)
                else:
                    fvars.append(v)
            return tvars, fvars

    def corr(self, c1, warp, lvl, scope='Correspondence'):
        """
        Build the cost volume for associating a cell from Voxel1 with its corresponding cell in Voxel2
        :param c1: The level of the feature pyramid of Voxel1
        :param warp: The warped level of the feature pyramid of Voxel2
        :param lvl: The level
        :return: cost volume
        """
        scope = f'{scope}{lvl}'
        with tf.name_scope(scope):
            return cost_volume(c1, warp, self.search_range, scope)

    def predict_flow_matching(self, c1, c2, up_flow, lvl, scope='Matching'):
        """
        Estimate optical flow, the first step.
        :param c1: B * D * H * W * C        The feature from Voxel1 at level lvl
        :param c2: B * D * H * W * C        The feature from Voxel2 at level lvl
        :param up_flow: B * D * H * W * 3   Predicted flow from the previous level but upsampled and magnified by 2
        """
        scope = f'{scope}{lvl}'
        with tf.variable_scope(scope):
            chan = 64
            # first compute cost volume
            if up_flow is None:
                up_flow = 0.
            else:
                c2 = self.warp(c2, up_flow)

            x = self.corr(c1, c2, lvl)

            # second compute flow
            with slim.arg_scope([slim.conv3d],
                                activation_fn=leaky_relu):
                x = slim.conv3d(x, chan//1, 3, 1, scope=f'conv{lvl}_1')
                x = slim.conv3d(x, chan//2, 3, 1, scope=f'conv{lvl}_2')
                x = slim.conv3d(x, chan//4, 3, 1, scope=f'conv{lvl}_3')

                delta = slim.conv3d(x, 3, 3, 1, activation_fn=None, scope=f'conv{lvl}_4',
                                    weights_initializer=tf.zeros_initializer(),
                                    biases_initializer=None)

            return delta + up_flow


class HairRNNNet(object):

    def __init__(self,
                 input_frames=3, prev_frames=2,                         # temporal information
                 input_dims=3, image_size=256,                          # input information
                 depth=96, height=128, width=128,                       # voxel information
                 latent_spatial_res=8,                                  # temporal information
                 scope_name="HairRNNNet"                                # model scope name
                 ):
        super().__init__()

        self.input_dims = input_dims
        self.image_size = image_size

        self.depth = depth
        self.height = height
        self.width = width

        self.min_channels = 16
        self.max_channels = 64

        self.latent_s = latent_spatial_res
        self.latent_d = self.depth // (self.height//self.latent_s)
        self.latent_c = self.max_channels
        assert(self.height == self.width)
        assert(self.latent_d > 0)

        self.spatial_ks = 3
        self.input_frames = input_frames
        self.prev_frames = prev_frames
        self.n_img_layers = np.int(np.log2(self.image_size // self.latent_s))
        self.n_vox_layers = np.int(np.log2(self.height // self.latent_s))
        assert(self.n_img_layers > 0)
        assert(self.n_vox_layers > 0)

        print("input_frames", self.input_frames)
        print("prev_frames", self.prev_frames)
        print("img_layers:", self.n_img_layers)
        print("vox_layers:", self.n_vox_layers)

        print("latent_D, latent_H, latent_W", self.latent_d, self.latent_s, self.latent_s)

        self.scope_name = scope_name

    def __call__(self, x, V):

        with tf.variable_scope(self.scope_name):

            caches = self.ImgEncoder(x)
            V = self.VoxEncoder(V)
            occ = self.decoder(caches.copy(), V, 1, scope="Decoder_For_Occ")
            ori = self.decoder(caches.copy(), V, 3, scope="Decoder_For_Ori")
            ori = pixel_norm(ori)
            occ = occ * 10.
            return occ, ori

    def ImgEncoder(self, x):
        # input N * 128 * 128 * C * T
        with tf.variable_scope("ImgEncoder"):

            with slim.arg_scope([slim.conv2d],
                                activation_fn=leaky_relu):

                caches = []
                # --------------------------- Down Sample -----------------------------#
                for i in range(self.n_img_layers):
                    num_outputs = min(self.max_channels, self.min_channels*(2**i))
                    x = slim.conv2d(x, num_outputs=num_outputs, kernel_size=self.spatial_ks, stride=2)
                    caches.append(x)

                return caches

    def VoxEncoder(self, V):
        # input N * 128 * 128 * C * T
        with tf.variable_scope("VoxEncoder"):

            with slim.arg_scope([slim.conv3d],
                                activation_fn=leaky_relu):

                for i in range(self.n_vox_layers):
                    num_outputs = min(self.max_channels, self.min_channels * (2 ** i))
                    V = slim.conv3d(V, num_outputs=num_outputs, kernel_size=self.spatial_ks, stride=2)

                return V

    def decoder(self, caches, V, out_channels, reuse=None, scope="Decoder"):
        caches = [self.convert_to_3D_feat(f) for f in caches]
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                                activation_fn=leaky_relu):
                # initialize
                C = self.latent_c
                V = self.to_Voxel3(caches.pop(), V, C, "ToVoxel0")
                O = slim.conv3d(V, out_channels, 1, 1, activation_fn=None, biases_initializer=None)

                # progressively learning
                num = len(caches)
                for i in range(num):
                    C //= 2
                    # key line
                    V = slim.conv3d_transpose(V, C, 4, 2)  # 4 & 2 is set to as norm, no overlapping, no keyboard effect
                    V = self.to_Voxel3(caches.pop(), V, C, scope=f"toVoxel{i+1}")

                    # second line
                    O = sample(O, 2.)
                    O += slim.conv3d(V, out_channels, 1, 1, activation_fn=None, biases_initializer=None)

                    print("voxel's shape: ", O.get_shape())
                return O

    def to_Voxel3(self, f, V, num_outputs, scope="ToVoxel"):
        """
        :param f:   converted 2D features
        :param V:   3D voxel feature upsampled from previous level
        :param G:   Global feature vector for this shape B * 1 * 1 * 1 * Cha
        :return:    the modified V with up sampling
        """
        assert f.shape.ndims == 5
        assert V.shape.ndims == 5

        with tf.variable_scope(scope):

            B = tf.shape(V)[0]
            _, D, H, W, _ = V.get_shape().as_list()
            P = get_spatial_points(B, D, H, W, True)

            with slim.arg_scope([slim.conv3d], kernel_size=3, activation_fn=leaky_relu):

                f = tf.tile(f, (1, D, 1, 1, 1))
                V = tf.concat([V, f, P], axis=-1)
                V = slim.conv3d(V, num_outputs)
                V = slim.conv3d(V, num_outputs)

                return V

    def convert_to_3D_feat(self, x):
        shape = x.get_shape().as_list()
        x = tf.reshape(x, [-1, 1, *shape[-3:]])
        return x


class Discriminator(object):
    def __init__(self,
                 image_size=256,  # input information
                 depth=96, height=128, width=128,  # voxel information
                 scope_name="Discriminator",
                 **kwargs):
        self.image_size = image_size

        self.depth = depth
        self.height = height
        self.width = width

        self.min_channels = kwargs.pop("min_channels", 16)
        self.max_channels = kwargs.pop("max_channels", 64)

        self.spatial_ks = 3
        self.latent_s = 4
        self.latent_d = self.depth // (self.height // self.latent_s)

        self.n_layers = int(np.log2(self.height // self.latent_s))
        assert (self.height == self.width)  # make sure that height is equal to width
        assert (self.latent_d > 0)          # make sure that depth of latent space is greater than zero
        assert (self.image_size % self.height == 0)  # make sure that image size are integer multiples of height

        self.scope_name = scope_name

    def __call__(self, inputs, fake_or_real, reuse=None):

        out = []
        with tf.variable_scope(self.scope_name, reuse=reuse):

            if inputs.shape.ndims == 5:
                shape = inputs.get_shape()[1:]
                if not shape.is_fully_defined():
                    raise ValueError('Inputs %s has undefined last dimensions %s.' % (inputs.name, shape))
                T, H, W, C = shape.as_list()

                inputs = tf.transpose(inputs, [0, 2, 3, 1, 4])
                inputs = tf.reshape(inputs, [-1, H, W, T * C])

            if self.image_size > self.height:
                inputs = tf.image.resize_nearest_neighbor(inputs, size=(self.height, self.height))

            inputs = to_Voxel(inputs, self.depth)
            x = tf.concat([inputs, fake_or_real], axis=-1)

            with slim.arg_scope([slim.conv3d],
                                padding="SAME",
                                activation_fn=leaky_relu):
                for i in range(self.n_layers):
                    C = min(self.max_channels, self.min_channels * (2**i))
                    x = slim.conv3d(x, C, 4, 2)
                    out.append(x)

                k = min(*x.get_shape().as_list()[-4:-1])
                x = slim.conv3d(x, 1, k, 1, padding="VALID", biases_initializer=None, activation_fn=None)
                print("Discriminator", x.get_shape())
                x = tf.reshape(x, [tf.shape(fake_or_real)[0], -1])
                x = tf.reduce_mean(x, axis=-1) * 0.01
                out.append(x)

            return out