import tensorflow as tf
import numpy as np
from Models.ops import nearest_resize_voxel, pixel_norm, warp_voxel, get_boundary
slim = tf.contrib.slim


def binary_cross_entropy(logits, labels, sample_ratio=1.0, scope="binary_cross_entropy_with_logits"):

    """
    NOTE: the range of the labels is {0, 1}
        r = gamma : to balance the training !!!
        z = labels
        x = logits
        loss =
        r * z * -log(sigmoid(x)) + (1 - r) * (1 - z) * -log(1 - sigmoid(x))
        = r * z * log(1 + exp(-x)) + (1 - r) * (1 - z) * (x + log(1 + exp(-x))
        = (1 - z - r + r * z) * x + (1 - z - r + 2 * r * z) * log(1 + exp(-x))
        set a = 1 - z - r
        set b = r * z
        (a + b) * x + (a + 2b) * log(1 + exp(-x))
        when x < 0, to prevent overflow
        (a + 2b) * log(1 + exp(-x)) = (a + 2b) * (-x + log(exp(x) + 1))

        when x < 0
        = - b * x + (a + 2b) * log(1 + exp(x))
        when x > 0
        = (a + b) * x + (a + 2b) * log(1 + exp(-x))

        to avoid overflow and enforce stability:
        = max(x, 0) * a + b * abs(x) + (a + 2b) * log(1 + exp(-abs(x))
    """

    with tf.name_scope(scope, values=[logits, labels]):

        logits = tf.convert_to_tensor(logits, name="logits")
        labels = tf.convert_to_tensor(labels, name="labels")
        weight = get_boundary(labels) * 49.0 + 1
        gamma = 0.85

        try:
            labels.get_shape().merge_with(logits.get_shape())
        except ValueError:
            raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                             (logits.get_shape(), labels.get_shape()))

        # labels and logits must be of the same type, and should not include gradient
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, logits.dtype)
        weight = tf.cast(weight, logits.dtype)
        labels = tf.stop_gradient(labels, name="labels_stop_gradient")

        # sample
        if sample_ratio < 1.:
            N, D, H, W, _ = [i.value for i in labels.get_shape()]
            logits = tf.reshape(logits, [N, -1])
            labels = tf.reshape(labels, [N, -1])
            weight = tf.reshape(weight, [N, -1])

            max_value = D * H * W
            sampled_cells = tf.random_uniform(shape=[int(max_value * sample_ratio)], minval=0, maxval=max_value, dtype=tf.int64)
            logits = tf.gather(logits, sampled_cells, axis=1)
            labels = tf.gather(labels, sampled_cells, axis=1)
            weight = tf.gather(weight, sampled_cells, axis=1)
            print("sampled_label_shape", labels.shape, labels.shape)

        a = (1 - gamma - labels)
        b = gamma * labels
        zeros = tf.zeros_like(logits, dtype = logits.dtype)
        cond = (logits >= zeros)
        relu_logits = tf.where(cond, logits, zeros)
        neg_abs_logits = tf.where(cond, -logits, logits)
        pos_abs_logits = tf.where(cond, logits, -logits)

        loss = weight * (a * relu_logits + b * pos_abs_logits + (a + 2 * b) * tf.log1p(tf.exp(neg_abs_logits)))

        return tf.reduce_sum(loss) / (tf.reduce_sum(weight) + 1e-8)


def laplacian_smooth3d(x, scope="laplacian_smooth"):

    """
    :param x: should have rank of 5  : N * D * H * W * C
    :return: the laplacian loss

    the second derivativs of x with respect to x,z,y,direction
    it can be used a 3D template, just like the 2D image convolution kernel
    """
    dim = x.get_shape()[-1:]
    if not dim.is_fully_defined():
        raise ValueError('Inputs %s has undefined last dimension %s.' % (x.name, dim))
    if not dim[0].value == 3:
        raise ValueError('Inputs %s has invalid last dimension, which should be 3.' % x.name)

    with tf.name_scope(scope, values=[x]):

        x = tf.convert_to_tensor(x, name="field")
        filters = np.array([[[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]],

                            [[0, 1, 0],
                             [1,-6, 1],
                             [0, 1, 0]],

                            [[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]]])

        filters = tf.convert_to_tensor(filters, name="filters", dtype = x.dtype)
        filters = tf.expand_dims(tf.expand_dims(filters, -1), -1) # should have the shape of D * H * W * 1 * 1

        x_x, x_y, x_z = tf.split(x, [1, 1, 1], axis = -1) # split the x, y, z part

        loss_x = tf.nn.conv3d(x_x, filters, [1, 1, 1, 1, 1], padding='SAME', data_format="NDHWC")
        loss_y = tf.nn.conv3d(x_y, filters, [1, 1, 1, 1, 1], padding='SAME', data_format="NDHWC")
        loss_z = tf.nn.conv3d(x_z, filters, [1, 1, 1, 1, 1], padding='SAME', data_format="NDHWC")

        return tf.reduce_mean(loss_x ** 2 + loss_y ** 2 + loss_z ** 2)


def uniform_sample_loss(gt_occ, gt_ori, out, sample_ratio=1.0):
    with tf.name_scope("uniform_sample_loss", values=[gt_occ, gt_ori, out]):
        if sample_ratio < 1.:
            N, D, H, W, _ = gt_occ.get_shape().as_list()
            gt_occ = tf.reshape(gt_occ, [N, -1, 1])
            gt_ori = tf.reshape(gt_ori, [N, -1, 3])
            out = tf.reshape(out, [N, -1, 3])

            max_value = D*H*W
            sampled_cells = tf.random_uniform(shape=[int(max_value*sample_ratio)], minval=0, maxval=max_value, dtype=tf.int64)
            gt_occ = tf.gather(gt_occ, sampled_cells, axis=1)
            gt_ori = tf.gather(gt_ori, sampled_cells, axis=1)
            out = tf.gather(out, sampled_cells, axis=1)
            print("sampled_gt_occ_shape", gt_occ.shape, gt_ori.shape)

        return l1_loss(gt_ori - out) / (tf.maximum(tf.reduce_sum(gt_occ), 1.0))


def l1_loss(x):

    return tf.reduce_sum(tf.abs(x))


def flow_loss(gt_o, gt_f, pr_f, sample_ratio=1.0):

    with tf.name_scope("flow_loss", values=[gt_o, gt_f, pr_f]):
        if isinstance(pr_f, (list, tuple)):
            loss = 0.
            weights = reversed([1/2**i for i in range(len(pr_f))])
            for w, f in zip(weights, pr_f):
                loss += flow_loss(gt_o, gt_f, f) * w
            return loss

        _, D, H, W, _ = gt_f.get_shape().as_list()
        _, d, h, w, _ = pr_f.get_shape().as_list()
        if d != D:
            ratio = float(h) / float(H)
            gt_o = nearest_resize_voxel(gt_o, shape=[d, h, w])
            gt_f = nearest_resize_voxel(gt_f, shape=[d, h, w]) * ratio  # should be scaled
        loss = uniform_sample_loss(gt_o, gt_f, pr_f * gt_o, sample_ratio=sample_ratio)
        return loss


def content_loss(gt_o, gt_f, pr_f):

    with tf.name_scope("content_loss", values=[gt_o, gt_f, pr_f]):

        _, D, H, W, _ = gt_o.get_shape().as_list()
        _, d, h, w, C = pr_f.get_shape().as_list()
        if d != D:
            gt_o = nearest_resize_voxel(gt_o, shape=[d, h, w])
            gt_o = tf.tile(gt_o, [1, 1, 1, 1, C])
        loss = uniform_sample_loss(gt_o, gt_f, pr_f)
        return loss


def adversarial_loss(netD, real, fake, inputs, masks):
    # inputs, real, fake should be in the same graph (values=[])
    with tf.name_scope("adversarial_loss", values=[inputs, real, fake]):
        alpha = tf.random_uniform(minval=0.0, maxval=1.0, shape=[tf.shape(real)[0], 1, 1, 1, 1])
        delta = alpha * fake + (1 - alpha) * real

        features_real = netD(inputs, real, reuse=None)
        features_fake = netD(inputs, fake, reuse=True)
        features_delta = netD(inputs, delta, reuse=True)

        gradient = tf.gradients(features_delta[-1], [delta])[0]
        gradient_norm = tf.norm(tf.layers.flatten(gradient * masks), axis=1)  # gradient penalty for WGAN

        G_loss = tf.reduce_mean(-features_fake[-1])
        G_FM = content_loss(masks, tf.stop_gradient(features_real[2]), features_fake[2])

        D_loss = tf.reduce_mean(features_fake[-1] - features_real[-1]) + 1e-3 * tf.reduce_mean(features_real[-1]**2)
        GP = 10.0 * tf.reduce_mean(tf.square(gradient_norm - 1.))

        return G_loss, G_FM, D_loss, GP

