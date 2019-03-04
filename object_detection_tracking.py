# This is a basic framework for object detection and tracking applied in a video using tensorflow for object detection
# and opencv for tracking

import argparse

import numpy as np
import os
import tensorflow as tf
import time
import cv2

import PIL.ImageOps

from lib_utils import filter_detections
import lib_utils as lu

__version__ = '0.1'
SCORE_THRESHOLD = 0.5
DETECTION_RATE = 5
MODEL_PATH = 'model/frozen_inference_graph.pb'
LABELS_MAP_PATH = 'model/label_map.pbtxt'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="ROBORDER Object detection")
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument("--visualize", action="store_true",
                        help="Whether to visualize the results in every frame. Default: False")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD,
                        help="The score above which bboxes are taken into consideration. "
                             "Default: {}".format(SCORE_THRESHOLD))
    parser.add_argument('-v', "--video-path", type=str,
                        help="The path to the video to be processed.")
    parser.add_argument('-m', "--model-path", type=str, default=MODEL_PATH,
                        help="Path to the frozen model to be used.")
    parser.add_argument('-l', "--labels-map-path", type=str, default=LABELS_MAP_PATH,
                        help="Path to the labels map to be used.")
    parser.add_argument("--detection-rate", type=int, default=DETECTION_RATE,
                        help="The rate of detection, it perform 1 detection every detection-rate value. "
                             "Default: {}".format(DETECTION_RATE))
    return parser.parse_args()


def load_image_into_numpy_array(image, inverse_image=False):
    (im_width, im_height) = image.size
    if image.mode == 'RGB':
        if inverse_image:
            image_np = np.array(PIL.ImageOps.invert(image).getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        else:
            image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
        return image_np
    else:
        if inverse_image:
            image_np = np.stack((np.squeeze(np.array(PIL.ImageOps.invert(image).getdata()).reshape((
                im_height, im_width, 1)).astype(np.uint8)),) * 3, axis=-1)
        else:
            image_np = np.stack((np.squeeze(np.array(image.getdata()).reshape((
                im_height, im_width, 1)).astype(np.uint8)),) * 3, axis=-1)
        return image_np


def prepare_video_reader_writer(video_in_path, video_out_path=None):
    if not video_out_path:
        vid_name, vid_ext = os.path.splitext(video_in_path)
        video_out_path = os.path.join(vid_name + '_detected' + vid_ext)
    video_in = cv2.VideoCapture(video_in_path)
    fourcc = cv2.VideoWriter_fourcc(*'DIV4')
    fps = video_in.get(cv2.CAP_PROP_FPS)
    h = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_out = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))
    return video_in, video_out


def main():
    if [int(x) for x in tf.__version__.split('.')] < [int(x) for x in '1.4.0'.split('.')]:
        raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

    args = get_arguments()
    video_path = args.video_path
    if not video_path:
        raise ValueError('Parameter --video-path is required for the module to run. '
                         'Please provide a path to a video to be processed!')
    score_threshold = args.score_threshold
    model_path = args.model_path
    labels_map_path = args.labels_map_path
    detection_rate = args.detection_rate
    visualize = args.visualize

    start = time.time()

    # ## Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    init_end = time.time()
    print("Elapsed time  of the initialization step was: {:3.2f} secs".format(init_end - start))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    video_in, video_out = prepare_video_reader_writer(video_path)
    categories = lu.get_labels(labels_map_path)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            fr = 0
            frames_from_detection = 0
            while True:
                ok, image = video_in.read()
                if not ok:
                    break
                if fr % 100 == 0:
                    print('Reading frame: {}...'.format(fr))

                if frames_from_detection == 0:
                    image_np = image[..., ::-1]
                    write_on = image_np.copy()
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    # Actual detection.
                    (boxes_det, scores, labels, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    boxes_det, scores, labels = filter_detections(boxes_det, scores, labels, image, score_threshold)
                    labels = [categories[x] for x in list(labels)]
                    title = '{} - Frame {} - detection'.format(os.path.basename(video_path), fr)
                    write_on = lu.display_bboxes_on_image_array_label(write_on, boxes_det, labels=labels, plot=visualize, title=title)
                    fr += 1
                    frames_from_detection = detection_rate - 1

                    tracker = cv2.MultiTracker_create()
                    for j in range(boxes_det.shape[0]):
                        # tracker needs x,y,w,h format
                        box = (boxes_det[j, 0], boxes_det[j, 1], boxes_det[j, 2] - boxes_det[j, 0],
                               boxes_det[j, 3] - boxes_det[j, 1])
                        ok = tracker.add(cv2.TrackerKCF_create(), image, box)
                        if not ok:
                            print('Adding bbox to tracker failed')

                else:
                    ok, image = video_in.read()
                    if not ok:
                        break
                    if fr % 100 == 0:
                        print('Reading frame: {}...'.format(fr))
                    image_np = image[..., ::-1]
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    write_on = image_np.copy()

                    ok, boxes_tr = tracker.update(image)
                    boxes_tr[:, 2] = boxes_tr[:, 0] + boxes_tr[:, 2]
                    boxes_tr[:, 3] = boxes_tr[:, 1] + boxes_tr[:, 3]

                    title = '{} - Frame {} - tracking'.format(os.path.basename(video_path), fr)
                    write_on = lu.display_bboxes_on_image_array_label(write_on, boxes_tr, labels=labels,
                                                                      plot=visualize, title=title)

                    fr += 1
                    frames_from_detection -= 1
                video_out.write(np.array(write_on)[..., ::-1])

    inference_elapsed = time.time()
    print("Inference_elapsed elapsed time was: {:3.2f} secs and {:3.2f} fps for {} images".format(inference_elapsed -
                                                                                                  init_end,
                                                                                                  fr /
                                                                                                  (inference_elapsed -
                                                                                                   init_end),
                                                                                                  fr))
    print("Total elapsed time was: {:3.2f} secs".format(inference_elapsed - start))


if __name__ == '__main__':
    main()
