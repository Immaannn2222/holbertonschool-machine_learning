#!/usr/bin/env python3
"""OBJECT DETECTION YOLOV3"""
import tensorflow.keras as K


class Yolo:
    """class Yolo that uses the Yolo v3 algorithm"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor"""
        self.model = K.models.load_model(filepath=model_path)
        f = open(classes_path, "r")
        self.class_names = [lines.split('\n')[0] for lines in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
