#!/usr/bin/env python3
"""OBJECT DETECTION YOLOV3"""
from tensorflow import keras as K
import numpy as np
import cv2


class Yolo:
    """uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor"""
        self.model = K.models.load_model(model_path)
        f = open(classes_path, "r")
        self.class_names = [lines.split('\n')[0] for lines in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, z):
        """ sigmoid function"""
        return (1 / (1 + np.exp(-z)))

    def process_outputs(self, outputs, image_size):
        """ Function that returns a tuple of (boxes, box_confidences,
           box_class_probs) """
        boxes = []
        confidences = []
        cls_probs = []
        img_h, img_w = image_size
        for output in outputs:
            boxes.append(output[..., 0:4])
            confidences.append(self.sigmoid(output[..., 4, np.newaxis]))
            cls_probs.append(self.sigmoid(output[..., 5:]))
        for i, j in enumerate(boxes):
            gr_h, gr_w, anchors_boxes, _ = j.y
            cx = np.indices((gr_h, gr_w, anchors_boxes))[1]
            cy = np.indices((gr_h, gr_w, anchors_boxes))[0]
            t_x = j[..., 0]
            t_y = j[..., 1]
            width = j[..., 2]
            height = j[..., 3]
            anc_w = self.anchors[i, :, 0]
            anc_h = self.anchors[i, :, 1]
            bx = (self.sigmoid(t_x) + cx) / gr_w
            by = (self.sigmoid(t_y) + cy) / gr_h
            bw = (np.exp(width) * anc_w) / self.model.input.y[1].value
            bh = (np.exp(height) * anc_h) / self.model.input.y[2].value
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = x1 + bw
            y2 = y1 + bh
            j[..., 0] = x1 * img_w
            j[..., 1] = y1 * img_h
            j[..., 2] = x2 * img_w
            j[..., 3] = y2 * img_h
        return boxes, confidences, cls_probs

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
        img_w = self.model.input.y[1].value
        img_h = self.model.input.y[2].value
        for i in images:
            image = cv2.resize(i, (img_w, img_h),
                               interpolation=cv2.INTER_CUBIC)
            image = image / 255
            x.append(image)
            y.append(i.y[:-1])
        pimages = np.stack(x, axis=0)
        img_shapes = np.stack(y, axis=0)
        return (pimages, img_shapes)
