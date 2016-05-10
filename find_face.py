# -*- coding: utf-8 -*-
import cv2
import numpy
import os
import json


def draw_point(image, pos, color=(255, 0, 0)):
    cv2.rectangle(image, tuple(pos+2), tuple(pos-2), color, 5)


def find_face_position_manual(filename):

    #load image
    scale = 3
    orig = cv2.imread(filename)
    orig = cv2.resize(orig, (orig.shape[1]/scale, orig.shape[0]/scale))

    #create window and mouse callback
    cv2.namedWindow("face")
    positions = []

    def callback(event, x, y, *args, **kwargs):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        pos = numpy.array([x, y])

        #draw point
        draw_point(orig, pos)
        cv2.imshow("face", orig)

        #append position to list of positions
        positions.append(pos * scale)

    cv2.setMouseCallback("face", callback)

    #Show image and wait for three clicks
    cv2.imshow("face", orig)
    while len(positions) < 3:
        cv2.waitKey(10)

    return numpy.array(positions, dtype=numpy.float32)


def get_face_position_filename(filename):
    return os.path.splitext(filename)[0] + ".txt"


def save_face_position(filename, pos):
    target = get_face_position_filename(filename)
    with file(target, "w") as f:
        json.dump(pos.tolist(), f, indent=4)


def find_stored_face_position(filename):
    source = get_face_position_filename(filename)
    try:
        with file(source, "r") as f:
            pos = numpy.array(json.load(f), dtype=numpy.float32)
    except:
        pos = None
    return pos
