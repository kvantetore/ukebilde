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

    return {
        "leye": positions[0],
        "reye": positions[1],
        "nose": positions[2],
        "landmarks": None,
    }


import dlib

shape_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_path)

LANDMARK_NOSE_TIP = 30

LANDMARK_JAW = numpy.r_[0:17]

LANDMARK_LEFT_BROW = numpy.r_[17:22]
LANDMARK_RIGHT_BROW = numpy.r_[22:27]
LANDMARK_BROW = numpy.r_[LANDMARK_LEFT_BROW, LANDMARK_RIGHT_BROW]

LANDMARK_NOSE_BASE = numpy.r_[27:31]
LANDMARK_NOSE_BOTTOM = numpy.r_[31:36]
LANDMARK_NOSE = numpy.r_[LANDMARK_NOSE_BASE, LANDMARK_NOSE_BOTTOM]

LANDMARK_LEFT_EYE = numpy.r_[36:42]
LANDMARK_RIGHT_EYE = numpy.r_[42:48]
LANDMARK_EYES = numpy.r_[LANDMARK_LEFT_EYE, LANDMARK_RIGHT_EYE]

LANDMARK_MOUTH_OUTER = numpy.r_[48:60]
LANDMARK_MOUTH_INNER = numpy.r_[60:68]
LANDMARK_MOUTH = numpy.r_[LANDMARK_MOUTH_OUTER, LANDMARK_MOUTH_INNER]

def find_face_position_in_image(scaled):
    for detector_scale in range(0, 4):
        faces = detector(scaled, detector_scale)
        if len(faces) == 1:
            break

    if len(faces) != 1:
        raise Exception("Cannot find face")

    shape = predictor(scaled, faces[0])
    points = numpy.array([[p.x, p.y] for p in shape.parts()], dtype=float)
    if points.shape != (68, 2):
        raise Exception("Unexpected number of face points " + points.shape)

    leye = numpy.average(points[LANDMARK_LEFT_EYE], axis=0)
    reye = numpy.average(points[LANDMARK_RIGHT_EYE], axis=0)
    nose = points[30, :]

    return {
        "leye": leye,
        "reye": reye,
        "nose": nose,
        "landmarks": points
    }


def find_face_position_automatic(filename, scale=3):
    print "Finding face in ", filename
    #load image
    orig = cv2.imread(filename)
    scaled = cv2.resize(orig, (orig.shape[1]/scale, orig.shape[0]/scale))

    pos = find_face_position_in_image(scaled)

    return {
        "leye": pos["eye"] * scale,
        "reye": pos["reye"] * scale,
        "nose": pos["nose"] * scale,
        "landmarks": pos["points"] * scale,
    }


def get_face_position_filename(filename):
    return os.path.splitext(filename)[0] + ".auto.txt"


def save_face_position(filename, pos):
    target = get_face_position_filename(filename)
    with file(target, "w") as f:
        json.dump({
            "leye": pos["leye"].tolist(),
            "reye": pos["reye"].tolist(),
            "nose": pos["nose"].tolist(),
            "landmarks": pos["landmarks"].tolist() if pos["landmarks"] is not None else None,
        }, f, indent=4)


def find_stored_face_position(filename):
    source = get_face_position_filename(filename)
    try:
        with file(source, "r") as f:
            data = json.load(f)
            pos = {
                "leye": numpy.array(data["leye"], dtype=float),
                "reye": numpy.array(data["reye"], dtype=float),
                "nose": numpy.array(data["nose"], dtype=float),
                "landmarks": numpy.array(data["landmarks"], dtype=numpy.float) if data["landmarks"] is not None else None
            }
    except Exception as ex:
        pos = None
    return pos
