import os
import cv2
import random
import scipy.io
import numpy as np
import struct

stepInv = 1. / 0.01015625
gridOrg = np.array([-0.65, -0.65, -0.4875], dtype=np.float32)


def get_mask(d, flip=False, image_size=256):
    mask = os.path.join(d, "Mask.png").replace("\\", "/")
    maskImg = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    maskImg = cv2.resize(maskImg, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    maskImg = np.expand_dims(maskImg, -1).astype(np.float32)

    if flip:
        maskImg = maskImg[:, ::-1, :]

    return np.ascontiguousarray(maskImg) / 255.0


def get_conditional_input_data(d, flip=False, random_noise=False, image_size=256):
    dep = os.path.join(d, "Depth.png").replace("\\", "/")
    ori = os.path.join(d, "OriSmooth2D.png").replace("\\", "/")

    depImg = cv2.imread(dep)
    oriImg = cv2.imread(ori)

    depImg = cv2.resize(depImg, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    oriImg = cv2.resize(oriImg, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    depData = depImg[:, :, -1:].astype(np.float32) / 255.0  # only R
    oriData = oriImg[:, :, [2, 1]].astype(np.float32) / 255.0  # R AND G

    masData = None

    W = depData.shape[1]

    if flip or random_noise:
        masData = get_mask(d, False, image_size)
        oriData = oriData * 2. - 1.

    if flip:
        masData = masData[:, ::-1, :]
        depData = depData[:, ::-1, :]
        oriData = oriData[:, ::-1, :] * np.array([-1., 1.])  # R should be flipped

    if random_noise:
        # print("random_noise")
        num_noises = 5
        max_window = 50
        random_pos = np.random.randint(0, W - max_window, size=[num_noises, 2])

        if random.random() > 0.5:
            noise = np.zeros(shape=[W, W], dtype=np.float32)

            for pos in random_pos:
                random_window = np.random.randint(10, max_window)
                random_window += 1 if random_window % 2 == 0 else 0
                if masData[pos[0], pos[1], 0] > 0.5:
                    wind = noise[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window]
                    h = wind.shape[0]
                    w = wind.shape[1]
                    gaussian_noise = np.random.normal(scale=1.0, size=(h, w))
                    gaussian_light = cv2.getGaussianKernel(h, h // 4) * cv2.getGaussianKernel(w, w // 4).transpose()
                    gaussian_light /= gaussian_light.max() + 1e-6
                    gaussian_light = np.clip(gaussian_light, 0, 1.0)
                    noise[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window] = gaussian_noise * gaussian_light * 0.5
                    num_noises -= 1

            if num_noises < 5:
                oriData += np.expand_dims(noise, -1)
                oriData /= np.sqrt(np.sum(oriData ** 2, axis=-1, keepdims=True)) + 1e-6
        else:
            max_kernel_size = 20
            min_kernel_size = 5
            for pos in random_pos:
                random_window = np.random.randint(20, max_window)
                random_window += 1 if random_window % 2 == 0 else 0
                if masData[pos[0], pos[1], 0] > 0.5:
                    size = np.random.randint(min_kernel_size, max_kernel_size)
                    wind = oriData[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window, :]

                    gaussian_light = cv2.getGaussianKernel(wind.shape[0], wind.shape[0] // 3) * cv2.getGaussianKernel(wind.shape[1], wind.shape[1] // 3).transpose()
                    gaussian_light /= gaussian_light.max() + 1e-6
                    gaussian_light = np.clip(gaussian_light, 0, 1.0)[:, :, np.newaxis]

                    oriData[pos[0]:pos[0] + random_window, pos[1]:pos[1] + random_window, :] = \
                        (cv2.blur(wind, (size, size)) * gaussian_light + wind * (1.0 - gaussian_light)) / (
                                    np.sqrt(np.sum(wind ** 2, axis=-1, keepdims=True)) + 1e-6)

    if flip or random_noise:
        oriData = (oriData + 1.0) * 0.5 * masData
        oriData = np.clip(oriData, 0., 1.)

    input = np.concatenate([depData, oriData], axis=-1)
    return np.ascontiguousarray(input)


def get_ground_truth_3D_occ(d, flip=False):
    file = os.path.join(d, "Occ3D.mat").replace("\\", "/")
    occ = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Occ'].astype(np.float32)
    occ = np.transpose(occ, [2, 0, 1])
    occ = np.expand_dims(occ, -1)  # D * H * W * 1

    if flip:
        occ = occ[:, :, ::-1, :]

    occ = np.ascontiguousarray(occ)
    return occ


def get_ground_truth_3D_ori(d, flip=False):
    file = os.path.join(d, "Ori3D.mat").replace("\\", "/")
    ori = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Ori'].astype(np.float32)
    ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, 96])
    ori = ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)

    if flip:
        ori = ori[:, :, ::-1, :] * np.array([-1.0, 1.0, 1.0])

    ori = np.ascontiguousarray(ori)
    return ori  # scaled


def get_ground_truth_forward(d, flip=False, normalize=False):
    file = os.path.join(d, "ForwardWarp.mat").replace("\\", "/")
    if os.path.exists(file):
        ori = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Warp'].astype(np.float32)
        ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, 96])
        ori = ori.transpose([0, 1, 3, 2]).transpose([2, 0, 1, 3])

        if flip:
            ori = ori[:, :, ::-1, :] * np.array([-1., 1., 1.])

        if normalize:
            ori = np.ascontiguousarray(ori) * stepInv
    else:
        ori = np.zeros(shape=[96, 128, 128, 3], dtype=np.float32)
    return ori * np.array([1., -1., -1.])  # be care of the coordinate


def get_ground_truth_bacward(d, flip=False, normalize=False):
    file = os.path.join(d, "BacwardWarp.mat").replace("\\", "/")
    if os.path.exists(file):
        ori = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Warp'].astype(np.float32)
        ori = np.reshape(ori, [ori.shape[0], ori.shape[1], 3, 96])  # to shape of H, W, C, D
        ori = ori.transpose([0, 1, 3, 2]).transpose(2, 0, 1, 3)  # to shape of H, W, D, C to shape of D, H, W, C

        if flip:
            ori = ori[:, :, ::-1, :] * np.array([-1., 1., 1.])

        if normalize:
            ori = np.ascontiguousarray(ori) * stepInv  # the voxel grid step
    else:
        ori = np.zeros(shape=[96, 128, 128, 3], dtype=np.float32)
    return ori * np.array([1., -1., -1.])  # be care of the coordinate


def get_ground_truth_hair_image(d, pt_num):
    file = os.path.join(d, "hair.hair").replace("\\", "/")
    with open(file, 'rb') as hair:
        nu = struct.unpack('i', hair.read(4))[0]
        nv = struct.unpack('i', hair.read(4))[0]
        nl = struct.unpack('i', hair.read(4))[0]
        nd = struct.unpack('i', hair.read(4))[0]

        hairUV = hair.read(nu * nv * nl * nd * 4)
        hairUV = struct.unpack('f' * nu * nv * nl * nd, hairUV)
        hairUV = np.array(hairUV, np.float32).reshape(nu, nv, nl, nd)
        if nl > pt_num:
            hairUV = hairUV[:, :, :pt_num, :]

        uvMask = np.linalg.norm(hairUV[:, :, 0, :], axis=-1)
        uvMask = (uvMask > 0).astype(np.float32)

        # note that we take the coordinates continuously instead of discretely, so 128 96
        hairUV -= gridOrg
        hairUV *= np.array([1., -1., -1.], dtype=np.float32) * stepInv
        hairUV += np.array([0, 128, 96], dtype=np.float32)

        hairUV = np.maximum(hairUV,
                            np.array([0, 0, 0], dtype=np.float32))  # note that voxels out of boundaries are minus
        hairUV = np.minimum(hairUV, np.array([127.9, 127.9, 95.9], dtype=np.float32))

        return uvMask, hairUV


def get_ground_truth_3D_dist(d, flip=False):
    file = os.path.join(d, "Dist3D.mat").replace("\\", "/")
    occ = scipy.io.loadmat(file, verify_compressed_data_integrity=False)['Dist'].astype(np.float32)
    occ = np.transpose(occ, [2, 0, 1])
    occ = np.expand_dims(occ, -1)  # D * H * W * 1

    if flip:
        occ = occ[:, :, ::-1, :]

    occ = np.ascontiguousarray(occ)
    return occ


def get_the_frames(d):
    frames = []
    for dd in os.listdir(d):
        if dd.startswith("frame"):
            frames.append(dd)

    frames.sort(key=lambda x: int(x[len("frame"):]))  # order the file name
    return frames


def get_the_videos(d):
    videos = {}
    for dir in os.listdir(d):
        if dir.startswith("video"):
            videos[dir] = get_the_frames(os.path.join(d, dir))

    return videos


class Video:

    def __init__(self, video_dir, frames):
        self.video_dir = video_dir
        self.frames = frames


def get_all_the_videos(dirs, interval=-1):
    if not isinstance(dirs, list):
        dirs = [dirs]

    videos = []
    for dir in dirs:
        vs = get_the_videos(dir)
        for name, frames in vs.items():
            if interval >= 3:
                # divide the video into many tiny videos based on the given interval
                video_num = max(len(frames) // interval, 1)
                for i in range(video_num):
                    begin = interval * i
                    end = interval * (i + 1) if i < video_num - 1 else len(frames)  # video
                    videos.append(Video(os.path.join(dir, name), frames[begin:end]))
            else:
                videos.append(Video(os.path.join(dir, name), frames))

    print("num of videos: {}".format(len(videos)))

    return videos


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)