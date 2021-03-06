#!/usr/bin/env python3
"""OBJECT DETECTION YOLOV3"""
import numpy as np
from tensorflow import keras as K
import cv2
import glob


class Yolo:
    """class Yolo that uses the Yolo v3 algorithm"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor"""
        self.model = K.models.load_model(model_path)
        f = open(classes_path, "r")
        self.class_names = [lines.split('\n')[0] for lines in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """implemnets the sigmoid function"""
        return 1 / (1 + np.exp(x))

    def process_outputs(self, outputs, image_size):
        """procees the output of a draknet model for a single image"""
        boxes = []
        box_conf = []
        cls_proba = []
        image_height, image_width = image_size
        for output in outputs:
            boxes.append(output[..., 0:4])
            box_conf.append(self.sigmoid(output[..., 4]))
            cls_proba.append(self.sigmoid(output[..., 5:]))
        for idx, b in enumerate(boxes):
            grid_h, grid_w, anchors_boxes, _ = b.shape
            cx = np.indices((grid_h, grid_w, anchors_boxes))[1]
            cy = np.indices((grid_h, grid_w, anchors_boxes))[0]
            tx = b[..., 0]
            ty = b[..., 1]
            width = b[..., 2]
            height = b[..., 3]
            anc_w = self.anchors[idx, :, 0]
            anc_h = self.anchors[idx, :, 1]
            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h
            bw = (np.exp(width) * anc_w) / self.model.input.shape[1].value
            bh = (np.exp(height) * anc_h) / self.model.input.shape[2].value
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            b[..., 0] = x1 * image_width
            b[..., 1] = y1 * image_height
            b[..., 2] = x2 * image_width
            b[..., 3] = y2 * image_height
        return boxes, box_conf, cls_proba

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filter boxes """

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """applies non max suppression"""

    @staticmethod
    def load_images(folder_path):
        """ imports images using cv"""
        imgs = []
        imgs_paths = []
        for path in glob.glob(folder_path + '/*'):
            imgs.append(cv2.imread(path))
            imgs_paths.append(path)
        return imgs, imgs_paths

    def preprocess_images(self, images):
        """Resize the images with inter-cubic interpolation
            Rescale all images to have pixel values in the range [0, 1] """
        x = []
        y = []
        img_w = self.model.input.shape[1].value
        img_h = self.model.input.shape[2].value
        for i in images:
            image = cv2.resize(i, (img_w, img_h),
                               interpolation=cv2.INTER_CUBIC)
            image = image / 255
            x.append(image)
            y.append(i.shape[:-1])
        pimages = np.stack(x, axis=0)
        img_shapes = np.stack(y, axis=0)
        return (pimages, img_shapes)
