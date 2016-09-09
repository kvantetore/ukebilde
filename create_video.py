from __future__ import division
import cv2
import os
import numpy
import datetime
from collections import namedtuple
from functools32 import lru_cache
from find_face import find_face_position_in_image, LANDMARK_JAW, LANDMARK_BROW, LANDMARK_NOSE
from skimage import transform
import math

import multiprocessing
import signal

out_folder = "./frames"

FramePart = namedtuple("FramePart", "filename weight")
Frame = namedtuple("Frame", "frame_index frame_parts date")


@lru_cache(maxsize=0)
def read_image(filename, video_size):
    #print "reading", filename
    img = cv2.imread(filename)

    scaled = cv2.resize(img, video_size)

    #find face landmarks
    pos = find_face_position_in_image(scaled)
    landmarks = pos["landmarks"]

    return scaled, landmarks


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


def blend_images(images, landmarks, weights):
    """
    Blends a set of images and landmarks according to its weights
    """
    if len(images) == 1:
        return images[0], landmarks[0]

    #normalize weights
    weights = weights / sum(weights)

    #create average pose
    landmarks_morph = morph_points(landmarks, weights)

    #warp and blend images
    shape = images[0][0].shape
    target = numpy.zeros(shape, dtype=float)
    for img, lm, w in zip(images, landmarks, weights):
        tfrm = transform.PiecewiseAffineTransform()
        tfrm.estimate(expand_points(img, landmarks_morph), expand_points(img, lm))
        img_warp = transform.warp(img, tfrm)

        target = target + img_warp * w

    return numpy.array(target * 255, dtype=numpy.uint8), landmarks_morph


def morph_image(img1, landmarks1, img2, landmarks2, weight):
    if weight == 0:
        return img1
    if weight == 1:
        return img2

    img_morph, landmarks_morph = blend_images([img1, img2], [landmarks1, landmarks2], [1 - weight, weight])
    return img_morph


def create_groups(lst, group_size):
    group_count = int(math.ceil(len(lst)/10.))
    return [lst[i*group_size:(i+1)*group_size] for i in range(group_count)]


def blend_groups(filenames, group_size=10, outliers=5, out_folder="./blend/"):
    groups = create_groups(filenames, group_size)
    for group in groups:
        #load all images in group
        images, landmarks = zip(*[read_image(f, (1280, 768)) for f in group])

        #find outliers by finding the landmarks which are most different from the others in the group
        dist = distance_matrix(landmarks)
        core_indexes = numpy.argsort(numpy.sum(dist, axis=0))[:-outliers]

        #filter out outliers
        images = [images[i] for i in core_indexes]
        landmarks = [landmarks[i] for i in core_indexes]

        #blend
        weights = numpy.ones(len(images))
        img_blend, landmarks_blend = blend_images(images, landmarks, weights)

        #write output
        _, filename = os.path.split(group[0])
        out_file = os.path.join(out_folder, filename)
        print out_file
        cv2.imwrite(out_file, img_blend)


DISTANCE_FEATURES = numpy.r_[LANDMARK_JAW, LANDMARK_BROW, LANDMARK_NOSE]


def distance(landmarks1, landmarks2):
    """
    Calculate the distance between two sets of landmarks. Returns a number
    indicating the difference in pose.
    """
    #return numpy.sum((landmarks1 - landmarks2)**2)
    dist = (landmarks1[DISTANCE_FEATURES, :] - landmarks2[DISTANCE_FEATURES, :])**2
    return numpy.average((numpy.sum(dist, axis=1)))


def distance_matrix(landmarks):
    """
    Calculate distance matrix between all pairs of landmkarks in a list
    """
    ret = numpy.zeros((len(landmarks), len(landmarks)), dtype=float)
    for i, l1 in enumerate(landmarks):
        for j, l2 in enumerate(landmarks):
                if j < i:
                    ret[j, i] = ret[i, j] = distance(l1, l2)
    return ret


class Timeline():
    def __init__(self, fps=24, video_size=(1280, 720)):
        self.fps = fps
        self.video_size = video_size
        self.frames = []

        self.start_date = datetime.datetime(2012, 8, 22)

    def create_frames_piecewise_blend(self, filenames, capture_dates, delay, fade):
        blend_frames = int(fade * self.fps)
        delay_frames = int(delay * self.fps)

        file_index = 0
        while file_index < len(filenames):
            #blend from previous file
            if file_index > 0:
                for i in xrange(blend_frames):
                    weight = i / float(blend_frames)
                    print weight
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
        frame_index = 0
        for image_index, (filename, capture_date) in enumerate(zip(filenames, capture_dates)):
            for sub_image_index in range(int(seconds_per_frame * self.fps)):
                sub_image_weight = sub_image_index / float(seconds_per_frame * self.fps - 1)

                #include window_size amount of images before and after the current image
                start = max(0, image_index - window_size + 1)
                end = min(image_index + window_size + 1, len(filenames))

                #use linear weights centered around the current frame (image index + sub image weight)
                indices = range(start, end)
                weights = numpy.array([window_size - (abs(x - image_index - sub_image_weight)) for x in indices])
                #weights /= sum(weights)
                print weights

                #create frame
                frame_parts = [FramePart(filenames[part_index], weight) for part_index, weight in zip(indices, weights)]
                self.frames.append(Frame(frame_index, frame_parts, capture_date))
                frame_index += 1

    def read_image(self, filename):
        return read_image(filename, self.video_size)

    def render_frame(self, frame):
        parts = frame.frame_parts

        #blend between frame parts
        sources, landmarks = zip(*[self.read_image(part.filename) for part in parts])
        weights = numpy.array([part.weight for part in parts])
        blend, _ = blend_images(sources, landmarks, weights)

        age = frame.date - self.start_date
        months = int(age.days / 30.4)

        if months > 0:
            cv2.putText(blend, "%imnd" % months, (int(blend.shape[1]/20), int(blend.shape[1]/15)), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 5)

        out_path = os.path.join(out_folder, "frame-{:0>5d}.jpg".format(frame.frame_index))
        print "writing", out_path

        cv2.imwrite(out_path, blend)

    def render(self, processes=8):
        if processes > 1:
            pool = multiprocessing.Pool(processes, init_worker)
            try:
                pool.map(render_frame, [(self, f) for f in self.frames])
            except KeyboardInterrupt:
                print "Caught KeyboardInterrupt, terminating workers"
                pool.terminate()
                pool.join()
        else:
            for frame in self.frames:
                self.render_frame(frame)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def render_frame((timeline, frame)):
    """
    hack around multiprocessing.Pools limitation on instancemethods
    """
    return timeline.render_frame(frame)


def render_frames(filenames, capture_dates, delay=.3, fade=0.2):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    timeline = Timeline()
    #timeline.create_frames_piecewise_blend(filenames, capture_dates, delay, fade)
    timeline.create_frames_sliding_blend(filenames, capture_dates, window_size=15, seconds_per_frame=.2)

    for f in timeline.frames:
        print "%s %f" % (f.frame_parts[0].filename, f.frame_parts[0].weight)

    timeline.render(processes=4)

def create_video():
    pass
