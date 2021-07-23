import tensorflow as tf
import numpy as np
from functools import reduce
slim = tf.contrib.slim
'''
def convolution(inputs,
                num_outputs,
                kernel_size,
                stride=1,
                padding='SAME',
                data_format=None,
                rate=1,
                activation_fn=nn.relu,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=initializers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=init_ops.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None,
                conv_dims=None):
'''


# ------------------------------ Operations -------------------------------- #
def leaky_relu(x, alpha=0.02):
    with tf.name_scope('LeakyRelu'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        return tf.maximum(x * alpha, x)


def pixel_norm(x, epsilon=1e-8):
    with tf.name_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True) + epsilon)  # rsqrt = 1./sqrt


def instance_normalize(x, epsilon=1e-8):
    with tf.name_scope('InstanceNorm'):
        inputs_rank = x.shape.ndims
        dim = x.get_shape()[-1:]
        if not dim.is_fully_defined():
            raise ValueError('Inputs %s has undefined last dimension %s.' % (x.name, dim))

        dim = dim[0].value
        beta = tf.get_variable(
            name='beta',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=True)
        scale = tf.get_variable(
            name='scale',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.ones_initializer(),
            trainable=True)

        moments_axes = list(range(1, inputs_rank - 1))
        mean, variance = tf.nn.moments(x, moments_axes, keep_dims=True)

        return tf.nn.batch_normalization(x, mean, variance, beta, scale, epsilon)


def instance_normalize2(x, epsilon=1e-8):

    B, T, H, W, C = x.get_shape().as_list()

    x = tf.reshape(x, [-1, H, W, C])
    x = instance_normalize(x, epsilon)
    x = tf.reshape(x, [B, T, H, W, C])

    return x


# ------------------------------ Modules -------------------------------- #
@slim.add_arg_scope
def partial_conv3d(inputs, msks,
                   num_outputs,
                   kernel_size,
                   stride,
                   activation_fn,
                   biases_initializer=tf.zeros_initializer(),
                   scope=None):
    """
    3D partial Convolution
    :param inputs:      B * D * H * W * C
    :param msks:        B * D * H * W * 1
    """
    with tf.variable_scope(scope):
        # normal convolution
        x = slim.conv3d(inputs, num_outputs, kernel_size, stride,
                        activation_fn=None, biases_initializer=None)

        # adjust masks
        m0 = slim.conv3d(msks, 1, kernel_size, stride, activation_fn=None,
                        weights_initializer=tf.ones_initializer(), biases_initializer=None,
                        trainable=False)

        # apply partial convolution
        if biases_initializer is not None:
            b = tf.get_variable(scope+"/bias", shape=[num_outputs], initializer=tf.initializers.zeros())
        else:
            b = 0

        t = m0 > 1e-6
        x = tf.where(t, (x * tf.to_float(kernel_size**3) / (m0+1e-6)) + b, 0)
        m = tf.where(t, 1., 0.)

        if activation_fn is not None:
            x = activation_fn(x)

        return x, m


@slim.add_arg_scope
def residual_block(x0, dilation=1, activation_fn=leaky_relu, normalizer_fn=None, scope="Residual"):

    dim = x0.get_shape()[-1:]
    if not dim.is_fully_defined():
        raise ValueError('Inputs %s has undefined last dimension %s.' % (x0.name, dim))

    dim = dim[0].value
    with tf.variable_scope(scope):

        with tf.contrib.slim.arg_scope([slim.conv2d],
                                       rate=dilation,
                                       padding="SAME",
                                       normalizer_fn=normalizer_fn):

            x1 = slim.conv2d(x0, dim, 3, scope="conv1", activation_fn=activation_fn)

            x2 = slim.conv2d(x1, dim, 3, scope="conv2", activation_fn=None)

            return x0 + x2


@slim.add_arg_scope
def residual_block3d(x0, dilation=1, activation_fn=leaky_relu, normalizer_fn=None, scope="Residual3D"):

    dim = x0.get_shape()[-1:]
    if not dim.is_fully_defined():
        raise ValueError('Inputs %s has undefined last dimension %s.' % (x0.name, dim))

    dim = dim[0].value

    with tf.variable_scope(scope):

        with tf.contrib.slim.arg_scope([slim.conv3d],
                                       rate=dilation,
                                       padding="SAME",
                                       normalizer_fn=normalizer_fn):

            x1 = slim.conv3d(x0, dim, 3, scope="conv1", activation_fn=activation_fn)

            x2 = slim.conv3d(x1, dim, 3, scope="conv2", activation_fn=None)

            return x0 + x2


@slim.add_arg_scope
def to_Voxel(x, depth, scope="ToVoxel"):

    shape = x.get_shape()[-3:]
    if not shape.is_fully_defined():
        raise ValueError('Inputs %s has undefined last dimensions %s.' % (x.name, shape))

    if x.shape.ndims >= 5:
        x = tf.reshape(x, [-1, *shape])

    with tf.variable_scope(scope):
        D = depth
        x = tf.expand_dims(x, 1)
        x = tf.tile(x, (1, D, 1, 1, 1))
        return x


@slim.add_arg_scope
def to_Voxel1(x, depth, num_outputs, scope="ToVoxel"):

    shape = x.get_shape()[-3:]
    if not shape.is_fully_defined():
        raise ValueError('Inputs %s has undefined last dimensions %s.' % (x.name, shape))

    if x.shape.ndims >= 5:
        x = tf.reshape(x, [-1, *shape])

    with tf.variable_scope(scope):
        D = depth
        _, H, W, _ = x.get_shape().as_list()
        # naive 
        with slim.arg_scope([slim.conv2d],
                            activation_fn=leaky_relu):
            x = slim.conv2d(x, D*num_outputs, 1, scope="conv1")  # expand channels, entails c_in*c_out*d parameters, a lot!
            x = tf.reshape(x, [-1, H, W, D, num_outputs])
            x = tf.transpose(x, [0, 3, 1, 2, 4])
        return x


@slim.add_arg_scope
def to_Voxel2(x, depth, num_outputs, up=None, scope="ToVoxel"):
    
    shape = x.get_shape()[-3:]
    if not shape.is_fully_defined():
        raise ValueError('Inputs %s has undefined last dimensions %s.' % (x.name, shape))

    if x.shape.ndims >= 5:
        x = tf.reshape(x, [-1, *shape])

    with tf.variable_scope(scope):
        D = depth
        _, H, W, _ = x.get_shape().as_list()
        
        p = get_spatial_points(x.shape[0], D, H, W)

        # first copy feature to each depth, then refine the feature according to p
        x = tf.expand_dims(x, 1)
        x = tf.tile(x, (1, D, 1, 1, 1))

        # idea borrowed from deep sdf
        x = tf.concat([x, up], axis=-1) if up is not None else x
        x = tf.concat([x, p], axis=-1)

        x = slim.conv3d(x, num_outputs, 1, activation_fn=leaky_relu)
        x = slim.conv3d(x, num_outputs, 1, activation_fn=leaky_relu)

        return x


# 3D Flow Network Related Operations
def get_grid_indices(D, H, W):

    x, _ = np.meshgrid(np.arange(W), np.arange(H))  # meshgrid(W, H) -> X (H*[0 ~ W]), Y ([0~H]*W);
    y, z = np.meshgrid(np.arange(H), np.arange(D))
    x = np.tile(np.reshape(x, [1, H, W]), [D, 1, 1])
    y = np.tile(np.reshape(y, [D, H, 1]), [1, 1, W])
    z = np.tile(np.reshape(z, [D, H, 1]), [1, 1, W])

    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)
    z = np.expand_dims(z, -1)

    grid = np.concatenate([x, y, z], axis=-1)[None]

    grid = tf.convert_to_tensor(grid)
    grid = tf.to_float(grid)
    return grid


def get_spatial_points(B, D, H, W, normalized=True):
    """
    :param B: batch size
    :param D: depth, should be specified first
    :param H: height, should be specified first
    :param W: width, should be specified first
    :param normalized: normalize the coordinates or not
    """
    xyz = get_grid_indices(D, H, W)
    xyz = tf.tile(xyz, [B, 1, 1, 1, 1])
    if normalized:
        xyz /= tf.to_float(np.array([W - 1, H - 1, D - 1]))
    return xyz


def get_voxel_value(voxel, z, y, x):
    """
    get voxel value having the shape of z's shape
    """

    B = tf.shape(z)[0]     # B is unknown
    S = z.get_shape()[1:]  # S must be known
    if not S.is_fully_defined():
        raise ValueError('Inputs %s has undefined last dimensions %s.' % (z.name, S))
    S = S.as_list()

    b = tf.range(0, B)
    for _ in S:
        b = tf.expand_dims(b, -1)
    b = tf.tile(b, [1] + S)

    indices = tf.stack([b, z, y, x], -1)
    indices = tf.stop_gradient(indices)

    return tf.gather_nd(voxel, indices) #indices.shape[:-1] + voxel.shape[indices.shape[-1]:]


def nearest_resize_voxel(voxel, shape):

    # voxel -> N * D * H * W * 1
    with tf.name_scope("NearestResizeVoxel"):
        def resize_by_axis(voxel, dim_1, dim_2, axis):

            resized_list = []
            unstack_list = tf.unstack(voxel, axis=axis)

            for img in unstack_list:
                resized_list.append(tf.image.resize_nearest_neighbor(img, [dim_1, dim_2]))

            return tf.stack(resized_list, axis=axis)

        ranks = len(voxel.get_shape())

        if ranks < 3:
            return

        if ranks == 3:
            voxel = tf.expand_dims(voxel, -1)
            voxel = tf.expand_dims(voxel, 0)

        if ranks == 4:
            voxel = tf.expand_dims(voxel, 0)

        assert(len(voxel.get_shape()) == 5)

        voxel = resize_by_axis(voxel, shape[1], shape[2], 1)
        voxel = resize_by_axis(voxel, shape[0], shape[1], 3)

        return voxel


def interpolation(V000, V001, V010, V011,
                  V100, V101, V110, V111,
                  wz, wy, wx, cal_normal=False):

    wzInv = 1. - wz
    wyInv = 1. - wy
    wxInv = 1. - wx

    i0 = V000 * wzInv + V100 * wz
    i1 = V010 * wzInv + V110 * wz

    j0 = V001 * wzInv + V101 * wz
    j1 = V011 * wzInv + V111 * wz

    v0 = i0 * wyInv + i1 * wy
    v1 = j0 * wyInv + j1 * wy

    v = v0 * wxInv + v1 * wx

    if not cal_normal:
        return v

    ll = v0  # left
    rr = v1  # right

    bb = i0 * wxInv + j0 * wx   # bottom
    uu = i1 * wxInv + j1 * wx   # up

    nn = (V000 * wyInv + V010 * wy) * wxInv + (V001 * wyInv + V011 * wy) * wx
    ff = (V100 * wyInv + V110 * wy) * wxInv + (V101 * wyInv + V111 * wy) * wx

    N = tf.concat([rr - ll, uu - bb, ff - nn], axis=-1) * -1.
    N = N / (tf.linalg.norm(N, axis=-1, keepdims=True) + 1e-6)

    return v, N


def linear_sample(voxel, nPos, warp_fn=get_voxel_value, sample_mode='Tri', D=96, H=128, W=128, cal_normal=False):
    with tf.name_scope("Sampling"):
        x, y, z = tf.unstack(nPos, num=3, axis=-1)
        maxZ = tf.cast(D - 1, tf.int32)
        maxY = tf.cast(H - 1, tf.int32)
        maxX = tf.cast(W - 1, tf.int32)
        zero = tf.zeros([], dtype=tf.int32)

        z0 = tf.cast(tf.floor(z), tf.int32)
        y0 = tf.cast(tf.floor(y), tf.int32)
        x0 = tf.cast(tf.floor(x), tf.int32)

        z0 = tf.clip_by_value(z0, zero, maxZ)
        y0 = tf.clip_by_value(y0, zero, maxY)
        x0 = tf.clip_by_value(x0, zero, maxX)

        if sample_mode == 'Random':
            z1 = z0 + tf.random_uniform(shape=z0.shape, maxval=2, dtype=tf.int32)
            y1 = y0 + tf.random_uniform(shape=z0.shape, maxval=2, dtype=tf.int32)
            x1 = x0 + tf.random_uniform(shape=z0.shape, maxval=2, dtype=tf.int32)

            z1 = tf.clip_by_value(z1, zero, maxZ)
            y1 = tf.clip_by_value(y1, zero, maxY)
            x1 = tf.clip_by_value(x1, zero, maxX)

            return warp_fn(voxel, z1, y1, x1)

        elif sample_mode == 'Tri':
            z1 = z0 + 1
            y1 = y0 + 1
            x1 = x0 + 1

            z1 = tf.clip_by_value(z1, zero, maxZ)
            y1 = tf.clip_by_value(y1, zero, maxY)
            x1 = tf.clip_by_value(x1, zero, maxX)

            V000 = warp_fn(voxel, z0, y0, x0)
            V001 = warp_fn(voxel, z0, y0, x1)
            V010 = warp_fn(voxel, z0, y1, x0)
            V011 = warp_fn(voxel, z0, y1, x1)

            V100 = warp_fn(voxel, z1, y0, x0)
            V101 = warp_fn(voxel, z1, y0, x1)
            V110 = warp_fn(voxel, z1, y1, x0)
            V111 = warp_fn(voxel, z1, y1, x1)

            wz = z - tf.to_float(z0)
            wy = y - tf.to_float(y0)
            wx = x - tf.to_float(x0)

            wz = tf.expand_dims(wz, -1)
            wy = tf.expand_dims(wy, -1)
            wx = tf.expand_dims(wx, -1)

            return interpolation(V000, V001, V010, V011,
                                 V100, V101, V110, V111,
                                 wz, wy, wx, cal_normal)
        else:
            raise ValueError("sample mode should be Tri or Random")


def sample(voxel, factor=2):

    with tf.name_scope(f"Res_to_{factor}x"):
        B, D, H, W, C = voxel.get_shape().as_list()  # as_list function returns value
        B = tf.shape(voxel)[0]                       # assume B is unknown
        grid = get_spatial_points(B, int(D*factor), int(H*factor), int(W*factor), False)
        nPos = grid / tf.to_float(factor)
        nPos = tf.stop_gradient(nPos)

        return linear_sample(voxel, nPos, D=D, H=H, W=W)


def partial_sample(voxel, mask, kernel_size, stride):

    with slim.arg_scope([slim.conv3d],
                        num_outputs=1,
                        kernel_size=kernel_size,
                        stride=stride,
                        activation_fn=None,
                        weights_initializer=tf.ones_initializer(),
                        biases_initializer=None,
                        trainable=False,
                        reuse=tf.AUTO_REUSE,
                        scope=f'FixedConv_{kernel_size}_{stride}'):

        m = slim.conv3d(mask)
        x, y, z = tf.unstack(voxel, num=3, axis=-1)
        x = slim.conv3d(tf.expand_dims(x, -1))
        y = slim.conv3d(tf.expand_dims(y, -1))
        z = slim.conv3d(tf.expand_dims(z, -1))
        v = tf.concat([x, y, z], axis=-1)

        v = v / (m+1e-6)

        return v, tf.to_float(m > 1e-6)


def close(voxel, kernel_size, stride=1):

    with slim.arg_scope([slim.conv3d],
                        num_outputs=1,
                        kernel_size=kernel_size,
                        stride=stride,
                        activation_fn=None,
                        weights_initializer=tf.ones_initializer(),
                        biases_initializer=None,
                        trainable=False,
                        reuse=tf.AUTO_REUSE,
                        scope=f'FixedConv_{kernel_size}_{stride}'):
        # dilate, padding must be the same
        voxel = slim.conv3d(voxel)
        voxel = tf.to_float(voxel > 0)

        # erode, padding must be the same
        voxel = slim.conv3d(voxel)
        voxel = tf.to_float(voxel >= kernel_size**3)

        return voxel


def close_ori(voxel, mask, kernel_size, stride=1):

    # first calculate the dilated region
    nmask = close(mask, kernel_size, stride)
    nmask = (1 - mask) * nmask  #xor

    with slim.arg_scope([slim.conv3d],
                        num_outputs=1,
                        kernel_size=kernel_size,
                        stride=stride,
                        activation_fn=None,
                        weights_initializer=tf.ones_initializer(),
                        biases_initializer=None,
                        trainable=False,
                        reuse=tf.AUTO_REUSE,
                        scope=f'FixedConv_{kernel_size}_{stride}'):

        # dilate the ori
        x, y, z = tf.unstack(voxel, num=3, axis=-1)
        x = slim.conv3d(tf.expand_dims(x, -1))
        y = slim.conv3d(tf.expand_dims(y, -1))
        z = slim.conv3d(tf.expand_dims(z, -1))
        nvoxel = tf.concat([x, y, z], axis=-1)
        nvoxel = pixel_norm(nvoxel)

        # blend
        ovoxel = nvoxel * nmask + voxel

        return ovoxel


def get_boundary(voxel):

    filters = np.array([[[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]],

                        [[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]],

                        [[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]])

    filters = tf.convert_to_tensor(filters, name="filters", dtype=voxel.dtype)
    filters = tf.expand_dims(tf.expand_dims(filters, -1), -1)  # should have the shape of D * H * W * 1 * 1

    boundes = tf.nn.conv3d(voxel, filters, [1, 1, 1, 1, 1], padding='SAME', data_format="NDHWC")
    boundes = tf.cast(tf.logical_and(boundes < 6, boundes > 0), voxel.dtype)

    return tf.stop_gradient(boundes)


def warp_voxel(voxel, flow, cal_normal=False):

    """
    :param voxel: of the shape [B, D, H, W, C]
    :param denseFlow: of the shape [B, D, H, W, 3]
    :return: warped voxel
    """

    B, D, H, W, C = voxel.get_shape().as_list()

    try:
        voxel.get_shape()[:-1].merge_with(flow.get_shape()[:-1])
    except ValueError:
        raise ValueError("voxel and flow must have the same shape (%s vs %s)" %
                         (voxel.get_shape(), flow.get_shape()))

    grid = get_grid_indices(D, H, W)
    nPos = grid + flow  # B * D * H * W * 3

    return linear_sample(voxel, nPos, D=D, H=H, W=W, cal_normal=cal_normal)


def cost_volume(voxel, warp, search_range, name):
    """Build cost volume for associating a pixel from Voxel1 with its corresponding cells in Voxel2.
    Args:
        c1: Level of the feature pyramid of Voxel1
        warp: Warped level of the feature pyramid of Voxel2
        search_range: Search range (maximum displacement), 2 is enough
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range], [search_range, search_range], [search_range, search_range], [0, 0]])
    _, D, H, W, _ = voxel.get_shape().as_list()
    max_offset = search_range * 2 + 1

    cost_vol = []
    for z in range(0, max_offset):
        for y in range(0, max_offset):
            for x in range(0, max_offset):
                slice = tf.slice(padded_lvl, [0, z, y, x, 0], [-1, D, H, W, -1])
                cost = tf.reduce_mean(voxel * slice, axis=-1, keepdims=True)
                cost_vol.append(cost)

    cost_vol = tf.concat(cost_vol, axis=-1)
    cost_vol = tf.nn.leaky_relu(cost_vol, alpha=0.2, name=name)

    return cost_vol


# ------------------------------ Models -------------------------------- #
class TemporalModelBase:

    def __init__(self, temporal_widths, channels, scope="TemporalModel"):

        for width in temporal_widths:
            assert (width % 2)

        assert (len(temporal_widths) == len(channels))

        self.n_layers = len(temporal_widths)

        self.temporal_widths = temporal_widths
        self.channels = channels
        self.scope = scope
        """
            k1 + 2a = k1*k2
            a = k1(k2 - 1) // 2
            assume k1, k2 are odds, the idea is to dilate-convolve the input sequence without overlaying.
            and the dilation of current layer equals to the receptive field of the previous layer 
        """

        k1 = 1
        self.pads = []
        for i in range(self.n_layers):
            k2 = temporal_widths[i]
            self.pads.append(k1 * (k2 - 1) // 2)
            k1 *= k2

        self.receptive_field = k1

        print("Receptive Field: ", self.receptive_field)
        print("Level pads: ", self.pads)

    def __call__(self, x, reuse=None):

        dim = x.get_shape()[1:2]
        if not dim.is_fully_defined():
            raise ValueError('Inputs %s has undefined second dimension %s.' % (x.name, dim))

        assert dim[0] >= self.receptive_field

        with tf.variable_scope(self.scope, reuse=reuse):
            return self._forward(x)

    def _forward(self, x):

        pass


class TemporalModel(TemporalModelBase):

    def __init__(self, spatial_kernel_size, temporal_widths, channels, scope="TemporalModel", dense=False):

        super().__init__(temporal_widths, channels, scope)

        assert (spatial_kernel_size % 2)
        self.spatial_ks = spatial_kernel_size
        self.dense = dense  # whether or not dense conv
        self.caches = []

    def _forward(self, x):
        """
        we use temporal convolution along the time coordinate
        :param x: with the shape of B * T * H * W * C,
        :return:
        """
        self.caches = []
        with slim.arg_scope([slim.conv3d], activation_fn=leaky_relu):

            for i in range(self.n_layers):
                # next_dilation actually equals to previous temporal width
                next_dilation = self.pads[i] // (self.temporal_widths[i] // 2)
                kernel_size = [self.temporal_widths[i] if not self.dense else self.pads[i] * 2 + 1, self.spatial_ks,
                               self.spatial_ks]
                rate = [next_dilation, 1, 1]
                print("kernel_size: ", kernel_size, "rate: ", rate)

                # first pad Height and Width
                s_pad = self.spatial_ks // 2
                x = tf.pad(x, paddings=[[0, 0], [0, 0], [s_pad, s_pad], [s_pad, s_pad], [0, 0]])
                x = slim.conv3d(x,
                                num_outputs=self.channels[i],
                                kernel_size=kernel_size,
                                padding="VALID",
                                rate=rate)
                x = slim.max_pool3d(x, [1, 2, 2], stride=[1, 2, 2])

                self.caches.append(x)

            # finally, only save the need
            T = self.caches[-1].get_shape().as_list()[1]
            for i in range(0, self.n_layers - 1):
                t = self.caches[i].get_shape().as_list()[1]
                begin = t // 2 - T // 2
                end = begin + T
                print(t, begin, end)
                self.caches[i] = self.caches[i][:, begin:end, :, :, :]

        return self.caches


class TemporalModelOptimized1f(TemporalModelBase):

    def __init__(self, spatial_kernel_size, temporal_widths, channels, scope="TemporalModel"):

        super().__init__(temporal_widths, channels, scope)

        assert (spatial_kernel_size % 2)
        self.spatial_ks = spatial_kernel_size
        self.caches = []

    def _forward(self, x):
        """
        we use temporal convolution along the time coordinate
        :param x: with the shape of B * T * H * W * C,
        :return:
        """
        self.caches = []
        with slim.arg_scope([slim.conv3d], activation_fn=leaky_relu):

            for i in range(self.n_layers):
                kernel_size = [self.temporal_widths[i], self.spatial_ks, self.spatial_ks]
                stride = [self.temporal_widths[i], 1, 1]
                print("kernel_size: ", kernel_size, "stride: ", stride)
                # first pad Height and Width
                s_pad = self.spatial_ks // 2
                x = tf.pad(x, paddings=[[0, 0], [0, 0], [s_pad, s_pad], [s_pad, s_pad], [0, 0]])
                x = slim.conv3d(x,
                                num_outputs=self.channels[i],
                                kernel_size=kernel_size,
                                stride=stride,
                                padding="VALID")
                x = slim.max_pool3d(x, [1, 2, 2], stride=[1, 2, 2])

                self.caches.append(x)

            T = self.caches[-1].get_shape().as_list()[1]
            assert (T == 1)
            for i in range(0, self.n_layers - 1):
                t = self.caches[i].get_shape().as_list()[1]
                begin = t // 2
                end = begin + T
                print(t, begin, end)
                self.caches[i] = self.caches[i][:, begin:end, :, :, :]

        return self.caches


class UNet:
    def __init__(self,
                 image_size=256,      # input information
                 depth=96, height=128, width=128,   # voxel information
                 temporal_widths=None,              # temporal information
                 scope_name="UNet",
                 **kwargs):

        self.image_size = image_size

        self.depth = depth
        self.height = height
        self.width = width

        self.min_channels = kwargs.pop("min_channels", 16)
        self.max_channels = kwargs.pop("max_channels", 64)

        self.spatial_ks = 3
        self.latent_s = 8
        self.latent_d = self.depth // (self.height // self.latent_s)
        self.layers_d = np.int(np.log2(self.image_size)) - np.int(np.log2(self.height)) - 1

        assert (self.layers_d >= 0)                     # make sure that image size is at least twice the height
        assert (self.height == self.width)              # make sure that height is equal to width
        assert (self.latent_d > 0)                      # make sure that depth of latent space is greater than zero
        assert (self.image_size % self.height == 0)     # make sure that image size are integer multiples of height

        self.scope_name = scope_name

        if isinstance(temporal_widths, list) and len(temporal_widths) > 0:
            self.temporalModel = True
            self.temporal_widths = temporal_widths
            self.receptive_field = reduce(lambda x, y: x * y, [1] + self.temporal_widths)
            self.n_layers = np.int(np.log2(self.image_size // self.latent_s) - len(self.temporal_widths))
            assert self.n_layers >= 0, "Incompatible settings caused by length of temporal widths"

            self.temporal_channels = []
            for i in range(len(self.temporal_widths)):
                self.temporal_channels.append(
                    min(self.max_channels, self.min_channels * (2 ** self.n_layers) * (2 ** i)))

            print("temporal_spatial_kernel_size:", self.spatial_ks)
            print("temporal_widths", self.temporal_widths)
            print("temporal_channels:", self.temporal_channels)

            self.temporal_subnet = TemporalModelOptimized1f(self.spatial_ks, self.temporal_widths,
                                                            self.temporal_channels)

        else:
            print("No Temporal Conv")
            self.temporalModel = False
            self.n_layers = np.int(np.log2(self.image_size // self.latent_s))
            assert self.n_layers >= 0, "Incompatible settings caused by input height"

    def __call__(self, x, reuse=None):
        pass

    def encoder(self, x, reuse=None):

        _, T, H, _, _ = x.get_shape().as_list()
        assert (H == self.image_size)

        # input N * T * 128 * 128 * C
        with tf.variable_scope("Encoder", reuse=reuse):
            with slim.arg_scope([slim.conv3d],
                                activation_fn=leaky_relu):
                caches = []
                # --------------------------- Down Sample -----------------------------#
                for i in range(self.n_layers):
                    num_outputs = min(self.max_channels, self.min_channels * (2 ** i))
                    x = slim.conv3d(x, num_outputs=num_outputs, kernel_size=[1, self.spatial_ks, self.spatial_ks], stride=[1, 2, 2])
                    caches.append(x[:, self.receptive_field // 2:T - self.receptive_field // 2, :, :, :])

                caches += self.temporal_subnet(x, reuse)

                return caches[self.layers_d:]

    def encoder_regular(self, x, reuse=None):

        if x.shape.ndims == 4:
            x = tf.expand_dims(x, 1)

        # input N * 1 * 128 * 128 * C
        with tf.variable_scope("Encoder", reuse=reuse):
            with slim.arg_scope([slim.conv3d],
                                activation_fn=leaky_relu):
                caches = []
                # --------------------------- Down Sample -----------------------------#
                for i in range(self.n_layers):
                    num_outputs = min(self.max_channels, self.min_channels * (2 ** i))
                    x = slim.conv3d(x, num_outputs=num_outputs, kernel_size=[1, self.spatial_ks, self.spatial_ks],
                                    stride=[1, 2, 2])
                    caches.append(x)

            return caches[self.layers_d:]

    def decoder(self, caches, out_channels, reuse=None, scope="Decoder"):

        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv3d],
                                activation_fn=leaky_relu):
                out = []
                num = len(caches) - 1   # since the res of the voxel is half of that of the image
                D, H, W = [self.latent_d, self.latent_s, self.latent_s]
                C = min(self.max_channels, self.min_channels * (2 ** num)) // 2     # reduce channels for 3D features

                v = to_Voxel1(caches.pop(), D, C, scope=f"toVoxel{num}")
                for i in reversed(range(num)):
                    # update
                    C = min(self.max_channels, self.min_channels * (2 ** i)) // 2
                    D *= 2

                    # key line
                    s = to_Voxel(caches.pop(), D, scope=f"toVoxel{i}")   # reduce parameters

                    # second line
                    v = slim.conv3d_transpose(v, C, 4, 2)
                    v = tf.concat([v, s], axis=-1)  # 12 * 16 * 16 * 64

                    # fine tune
                    v = slim.conv3d(v, C, 3, 1)  # 12 * 16 * 16 * 32
                    v = slim.conv3d(v, C, 3, 1)  # 12 * 16 * 16 * 32

                    o = slim.conv3d(v, out_channels, 3, 1, activation_fn=None, biases_initializer=None)
                    out.append(o)

                return out

    def decoder1(self, caches, out_channels, reuse=None, scope="Decoder"):

        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv3d],
                                activation_fn=leaky_relu):

                num = len(caches) - 1
                D, H, W = [self.latent_d, self.latent_s, self.latent_s]
                C = min(self.max_channels, self.min_channels * (2 ** num)) // 2     # reduce channels for 3D features

                v = to_Voxel1(caches.pop(), D, C, scope=f"toVoxel{num}")
                for i in reversed(range(num)):
                    # update
                    C = min(self.max_channels, self.min_channels * (2 ** i)) // 2
                    D *= 2

                    # key line
                    s = to_Voxel1(caches.pop(), D, C, scope=f"toVoxel{i}")    # need a lot of parameters, not recommended

                    # second line
                    v = slim.conv3d_transpose(v, C, 4, 2)
                    v = tf.concat([v, s], axis=-1)  # 12 * 16 * 16 * 64

                    # fine tune
                    v = slim.conv3d(v, C, 3, 1)  # 12 * 16 * 16 * 32
                    v = slim.conv3d(v, C, 3, 1)  # 12 * 16 * 16 * 32

                v = slim.conv3d(v, out_channels, 3, 1, activation_fn=None, biases_initializer=None)
                return v

    def decoder2(self, caches, out_channels, reuse=None, scope="Decoder"):

        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                                activation_fn=leaky_relu):

                num = len(caches) - 1
                D = self.latent_d
                C = min(self.max_channels, self.min_channels * (2 ** num)) // 2     # reduce channels for 3D features

                s = to_Voxel2(caches.pop(), D, C, None, scope=f"toVoxel{num}")
                v = slim.conv3d(s, out_channels, 1, 1, activation_fn=None, biases_initializer=None)
                for i in reversed(range(num)):
                    # update
                    D *= 2
                    C = min(self.max_channels, self.min_channels * (2 ** i)) // 2

                    # key line
                    s = slim.conv3d_transpose(s, C, 4, 2)  # 4 & 2 is set to as norm, no overlapping, no keyboard effect
                    s = to_Voxel2(caches.pop(), D, C, s, scope=f"toVoxel{i}")     # deep sdf

                    # second line
                    v = sample(v, 2)
                    v += slim.conv3d(s, out_channels, 3, 1, activation_fn=None, biases_initializer=None)

                return v
