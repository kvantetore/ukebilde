from __future__ import division
import cv2
import os
import numpy
import datetime
import dateutil.parser
from collections import namedtuple
from functools import lru_cache, wraps
from skimage import transform
import math
from itertools import groupby
import av

import multiprocessing
import signal

from find_face import find_face_position_in_image, LANDMARK_JAW, LANDMARK_BROW, LANDMARK_NOSE
from render_gpu import GpuRenderer


out_folder = "./frames"

FramePart = namedtuple("FramePart", "filename weight")
Frame = namedtuple("Frame", "frame_index frame_parts date")

@lru_cache(maxsize=1000)
def read_image(filename, video_size):
    print("reading", filename)
    img = cv2.imread(filename)

    scaled = cv2.resize(img, video_size)

    #find face landmarks
    pos = find_face_position_in_image(scaled)
    landmarks = pos["landmarks"]

    return scaled, expand_points(scaled, landmarks)


def morph_points(points, weights):
    ret = None
    for p, w in zip(points, weights):
        if ret is None:
            ret = p * w
        else:
            ret = ret + p * w

    return ret


def expand_points(img, points):
    """
    Expand landmark points to include the four corners of the image,
    in order to get the pixels outside the face included in the warped image
    """
    point_list = points.tolist()
    point_list.append([0, 0])
    point_list.append([img.shape[1], 0])
    point_list.append([img.shape[1], img.shape[0]])
    point_list.append([0, img.shape[0]])
    return numpy.array(point_list)


def blend_images(images, landmarks, weights, landmarks_morph):
    """
    Blends a set of images and landmarks according to its weights
    """
    if len(images) == 1:
        return images[0]

    #warp and blend images
    shape = images[0][0].shape
    target = numpy.zeros(shape, dtype=float)
    for img, lm, w in zip(images, landmarks, weights):
        tfrm = transform.PiecewiseAffineTransform()
        tfrm.estimate(landmarks_morph, lm)
        img_warp = transform.warp(img, tfrm)

        target = target + img_warp * w

    return numpy.array(target * 255, dtype=numpy.uint8)


class Timeline():
    def __init__(self, start_date, fps=24, video_size=(1280, 720), renderer=None):
        self.fps = fps
        self.video_size = video_size
        self.frames = []
        self.start_date = start_date
        
        if renderer != None:
            self.renderer = renderer
        else:
            self.renderer = blend_images


    def create_frames_piecewise_blend(self, filenames, capture_dates, delay, fade):
        blend_frames = int(fade * self.fps)
        delay_frames = int(delay * self.fps)

        file_index = 0
        while file_index < len(filenames):
            #blend from previous file
            if file_index > 0:
                for i in xrange(blend_frames):
                    weight = i / float(blend_frames)
                    print(weight)
                    frame = Frame(len(self.frames), [
                        FramePart(filenames[file_index-1], 1 - weight),
                        FramePart(filenames[file_index], weight),
                    ], capture_dates[file_index-1])
                    self.frames.append(frame)

            for i in xrange(delay_frames):
                frame = Frame(len(self.frames), [
                    FramePart(filenames[file_index], 1.0),
                ], capture_dates[file_index])
                self.frames.append(frame)

            file_index += 1

    def create_frames_sliding_blend(self, filenames, capture_dates, window_size=5, seconds_per_frame=1):
        """
        Create frames blending between images, at each frame taking into account window_size images before and after
         the current image
        """

        #we duplicate the filenames at the beginning and end to start and end on a "pure" image
        filenames = ([filenames[0]] * int(self.fps * seconds_per_frame)) + \
                    ([filenames[0]] * window_size) +                \
                    filenames +                                     \
                    ([filenames[-1]] * window_size) +               \
                    ([filenames[-1]] * int(self.fps * seconds_per_frame))

        capture_dates = ([capture_dates[0]] * int(self.fps * seconds_per_frame)) + \
                        ([capture_dates[0]] * window_size) +                \
                        capture_dates +                                     \
                        ([capture_dates[-1]] * window_size) +               \
                        ([capture_dates[-1]] * int(self.fps * seconds_per_frame))

        frame_index = 0
        for image_index, (filename, capture_date) in enumerate(zip(filenames, capture_dates)):
            sub_frame_count = max(1, int(seconds_per_frame * self.fps))
            for sub_image_index in range(sub_frame_count):                
                sub_image_weight = 0 if sub_frame_count == 1 else sub_image_index / float(sub_frame_count - 1)

                #include window_size amount of images before and after the current image
                start = max(0, image_index - window_size + 1)
                end = min(image_index + window_size + 1, len(filenames))

                #use linear weights centered around the current frame (image index + sub image weight)
                indices = range(start, end)
                weights = numpy.array([window_size - (abs(x - image_index - sub_image_weight)) for x in indices], dtype=float)
                weights /= sum(weights)

                #group parts by filename
                grouped_parts = groupby(zip(indices, weights), lambda x: filenames[x[0]])
                frame_parts = [FramePart(filename, min(window_size, sum([x[1] for x in p]))) for filename, p in grouped_parts]

                #create frame
                self.frames.append(Frame(frame_index, frame_parts, capture_date))
                frame_index += 1

    def read_image(self, filename):
        return read_image(filename, self.video_size)

    def render_frame(self, frame, container, stream):
        parts = frame.frame_parts

        #blend between frame parts
        sources, landmarks = zip(*[self.read_image(part.filename) for part in parts])
        weights = numpy.array([part.weight for part in parts])

        #create average pose
        landmarks_morph = morph_points(landmarks, weights)

        blend = self.renderer(sources, landmarks, weights, landmarks_morph)

        age = frame.date - self.start_date
        months = int(age.days / 30.4)

        if months > 0:
            print("%imnd" % (months, ))
            blend = cv2.putText(blend, "%imnd" % months, (int(blend.shape[1]/20), int(blend.shape[1]/15)), cv2.FONT_HERSHEY_SIMPLEX, 2, (25, 25, 25), 5)

        out_path = os.path.join(out_folder, "frame-{:0>5d}.jpg".format(frame.frame_index))
        print("writing frame", frame.frame_index)
        #cv2.imwrite(out_path, blend)

        video_frame = av.VideoFrame.from_ndarray(blend, format='bgra')
        for packet in stream.encode(video_frame):
           container.mux(packet)

    def render(self, output):
        container = av.open(output, mode='w')
        stream = container.add_stream('h264', rate=self.fps)
        stream.width = self.video_size[0]
        stream.height = self.video_size[1]
        stream.pix_fmt = 'yuv420p'

        for frame in self.frames:
            self.render_frame(frame, container, stream)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

        container.close()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def render_frame(input):
    """
    hack around multiprocessing.Pools limitation on instancemethods
    """
    timeline, frame = input
    return timeline.render_frame(frame)


def render_frames(filenames, capture_dates, settings):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    start_date = dateutil.parser.parse(settings["start_date"])
    fps = int(settings.get("fps", 24))
    video_height = int(settings.get("video_height", 720))
    video_width = int(settings.get("video_width", video_height * 16 / 9))
    output = settings.get("output", "ukebile.avi")

    window_size = int(settings.get("window_size", 5))
    seconds_per_frame = float(settings.get("seconds_per_frame", 0.3))

    renderer = GpuRenderer((video_width, video_height))
    timeline = Timeline(start_date=start_date, fps=fps, video_size=(video_width, video_height), renderer=renderer.morph_images)
    timeline.create_frames_sliding_blend(filenames, capture_dates, window_size=window_size, seconds_per_frame=seconds_per_frame)

    for f in timeline.frames:
        print("%s %f" % (f.frame_parts[0].filename, f.frame_parts[0].weight))

    timeline.render(output)

def create_video():
    pass
