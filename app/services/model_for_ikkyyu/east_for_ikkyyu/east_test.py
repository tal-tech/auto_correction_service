# -*- coding: utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
import shutil

import lib.lanms.adaptor as lanms
import lib.lanms.locality_aware_nms as nms_locality

import lib.network.model as model

from lib.dataset.dataload import restore_rectangle
from lib.utils.utils import cfg
import tqdm


class EastTest(object):

    def inference(self):
        """
        推理阶段
        :return:
        """

        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # 调用模型函数,返回分数和几何
        self.f_score, self.f_geometry = model.model(self.input_images, is_training=False)

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 采用滑动平均的方法更新参数,衰减速率（decay），用于控制模型的更新速度,迭代的次数
        # 指数加权平均的求法，公式 total = a * total + (1 - a) * next
        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        # 保存和恢复都需要实例化一个Saver
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        # 添加对显存的限制 add by boby 20190603
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.sess = tf.Session(config=gpu_config)

        # self.sess = tf.Session()

        # 函数通过checkpoint文件找到模型文件名
        ckpt = tf.train.get_checkpoint_state(cfg["TEST"]["CHECKPOINT_DIR"])
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')

        # 重载模型的参数，继续训练或用于测试数据
        saver.restore(self.sess, ckpt.model_checkpoint_path)
        print("Done")

    def test(self, img_path):

        # 将cv读取的BGR->RGB
        img = cv2.imread(img_path)[:, :, ::-1]

        resized_img, (ratio_h, ratio_w) = self.__resize_img(img)

        timer = {'net': 0, 'restore': 0, 'nms': 0}
        # 返回当前时间的时间戳
        start = time.time()

        score, geometry = self.sess.run([self.f_score, self.f_geometry], feed_dict={self.input_images: [resized_img]})

        timer['net'] = time.time() - start

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('{} : '.format(img_path))

        # 检测
        boxes, score_rgb_map, timer = self.__detect(score_map=score, geo_map=geometry, timer=timer)

        # 打印执行时间
        print('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(timer['net'] * 1000, timer['restore'] * 1000,
                                                                    timer['nms'] * 1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start
        print('[timing] {}'.format(duration))

        if boxes is not None:
            res_file = os.path.join(
                cfg["TEST"]["RESULT_TXT_DIR"],
                '{}.txt'.format(os.path.basename(img_path).split('.')[0]))

            with open(res_file, 'w') as f:
                for box in boxes:
                    # to avoid submitting errors
                    box = self.__sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1],
                                                                 box[1, 0], box[1, 1],
                                                                 box[2, 0], box[2, 1],
                                                                 box[3, 0], box[3, 1], ))
                    cv2.polylines(img[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))],
                                  True, color=(0, 255, 0), thickness=2)

        if cfg["TEST"]["WRITE_IMAGES"]:
            img_path = os.path.join(cfg["TEST"]["RESULT_DIR"], os.path.basename(img_path))
            cv2.imwrite(img_path, img[:, :, ::-1])

            img_path = os.path.join(cfg["TEST"]["SCORE_MAP_DIR"], os.path.basename(img_path))
            cv2.imwrite(img_path, score_rgb_map)

            geometry = geometry * score

            h, w, _ = geometry[0].shape

            geo1_map = np.concatenate(
                (geometry[0][:, :, 3].reshape((h, w, 1)), geometry[0][:, :, 0].reshape((h, w, 1)), score[0]), axis=2)
            img_path = os.path.join(cfg["TEST"]["GEO_MAP_DIR"], "1_" + os.path.basename(img_path))
            cv2.imwrite(img_path, geo1_map)

            geo2_map = np.concatenate(
                (geometry[0][:, :, 2].reshape((h, w, 1)), geometry[0][:, :, 1].reshape((h, w, 1)), score[0]), axis=2)
            img_path = os.path.join(cfg["TEST"]["GEO_MAP_DIR"], "2_" + os.path.basename(img_path))
            cv2.imwrite(img_path, geo2_map)

            thred_map = np.concatenate(
                (geometry[0][:, :, 4].reshape((h, w, 1)), geometry[0][:, :, 4].reshape((h, w, 1)),
                 geometry[0][:, :, 4].reshape((h, w, 1))), axis=2)
            img_path = os.path.join(cfg["TEST"]["GEO_MAP_DIR"], "3_" + os.path.basename(img_path))
            cv2.imwrite(img_path, thred_map * 255)

    # 试验发现score map threshold两级分化,所以阈值影响不大
    # def __detect(self, score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    def __detect(self, score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.01, nms_thres=0.2):
        """
        restore text boxes from score map and geo map
        :param score_map:
        :param geo_map:
        :param timer:
        :param score_map_thresh: threshhold for score map
        :param box_thresh: threshhold for boxes
        :param nms_thres: threshold for nms
        :return:
        """
        # input()
        if len(score_map.shape) == 4:
            score_map = score_map[0, :, :, 0]
            geo_map = geo_map[0, :, :, ]

        score_rgb_map = cv2.cvtColor(score_map * 255, cv2.COLOR_GRAY2RGB)
        # filter the score map
        xy_text = np.argwhere(score_map > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        start = time.time()
        # (xy_text[:, ::-1] * 4).shape [n,2] 乘以4原因是将图片缩放到512
        # geo_map[xy_text[:, 0], xy_text[:, 1], :] 得到过滤后的每个像素点的坐标和角度
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        # text_box_restored shape = [N, 4,2]
        print('{} text boxes before nms'.format(text_box_restored.shape[0]))
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        timer['restore'] = time.time() - start
        # nms part
        start = time.time()
        boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
        boxes = np.array(boxes)
        print('{} text boxes after nms'.format(boxes.shape[0]))
        timer['nms'] = time.time() - start

        if boxes.shape[0] == 0:
            return None, None, timer

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]

        return boxes, score_rgb_map, timer

    def __resize_img(self, img, max_side_len=2400):
        """
        将图像进行缩放,最大边大于2400,按照最大边进行resize,然后判断每个边是否能够被32整除,再进行一次resize
        :param im:
        :param max_side_len:
        :return:
        """
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        # limit the max side
        if max(resize_h, resize_w) > max_side_len:
            ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
        else:
            ratio = 1.
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
        resized_img = cv2.resize(img, (int(resize_w), int(resize_h)))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return resized_img, (ratio_h, ratio_w)

    def __sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]


if __name__ == "__main__":

    east = EastTest()
    east.inference()
    # 测试图片文件夹路径
    img_dir = cfg["TEST"]["TEST_DIR"]
    # 读取文件夹下路径
    img_list = os.listdir(img_dir)

    # 结果图文件夹
    if os.path.exists(cfg["TEST"]["RESULT_DIR"]):
        shutil.rmtree(cfg["TEST"]["RESULT_DIR"])
    os.makedirs(cfg["TEST"]["RESULT_DIR"])

    if os.path.exists(cfg["TEST"]["RESULT_TXT_DIR"]):
        shutil.rmtree(cfg["TEST"]["RESULT_TXT_DIR"])
    os.makedirs(cfg["TEST"]["RESULT_TXT_DIR"])

    # 得分图路径
    if os.path.exists(cfg["TEST"]["SCORE_MAP_DIR"]):
        shutil.rmtree(cfg["TEST"]["SCORE_MAP_DIR"])
    os.makedirs(cfg["TEST"]["SCORE_MAP_DIR"])

    # 开始测试
    for i in tqdm.tqdm(img_list):
        east.test(os.path.join(img_dir, i))
