# coding=utf-8
import PIL
from PIL import Image, ImageDraw, ImageFont
import os
from matplotlib import pyplot as plt
# from object_detection.utils.visualization_utils import draw_bounding_box_on_image
import numpy as np
import re


def display_bboxes_on_image_array_label(image_source, bboxes, labels=None, plot=True, title=''):
    """
    Display the bboxes over the image using an array as parameter plus providing a label for the objects
    :param title:
    :param labels:
    :param bboxes:
    :param image_source:
    :param plot:
    :return:
    """
    display_labels = False
    path_label = False
    if labels is not None:
        display_labels = True
    if isinstance(image_source, PIL.Image.Image):
        image = image_source
    elif isinstance(image_source, np.ndarray):
        image = Image.fromarray(image_source)
    elif os.path.exists(image_source):
        image = Image.open(image_source)
        path_label = True
    colors = ['red', 'blue', 'green', 'black', 'orange']
    color_id = 0
    class_color_dict = {}
    for i in range(bboxes.shape[0]):
        if display_labels:
            display_str_list = [labels[i]]
            if labels[i] in class_color_dict:
                color = class_color_dict[labels[i]]
            else:
                class_color_dict[labels[i]] = colors[color_id % len(colors)]
                color = colors[color_id % len(colors)]
                color_id += 1
        else:
            display_str_list = ['']
            color = 'red'
        # print('Color: {}, color_id={}, label={}'.format(color, color_id, display_str_list[0]))
        draw_bounding_box_on_image(image, bboxes[i, 1], bboxes[i, 0], bboxes[i, 3], bboxes[i, 2],
                                   color=color, thickness=2, display_str_list=display_str_list,
                                   use_normalized_coordinates=False)

    if plot:
        if path_label and not title:
            plt.figure(os.path.basename(image_source), figsize=(16, 12), dpi=80)
        else:
            plt.figure(title, figsize=(16, 12), dpi=80)
        plt.imshow(image)
        plt.waitforbuttonpress()
        plt.close()
    return image


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def filter_detections(boxes_det, scores, classes, image, score_thres=0.5):
    boxes_det = np.squeeze(boxes_det)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)
    h, w, _ = image.shape
    res = np.where(scores > score_thres)
    if not res[0].shape[0]:
        boxes_det = np.zeros((0, 4))
        scores = np.zeros((0, 1))
        classes = np.zeros((0, 1))
        return boxes_det, scores, classes
    n = np.where(scores > score_thres)[0][-1] + 1

    # this creates an array with just enough rows as object with score above the threshold
    # format: absolute x, y, x, y
    boxes_det = np.array([boxes_det[:n, 1] * w, boxes_det[:n, 0] * h, boxes_det[:n, 3] * w, boxes_det[:n, 2] * h]).T
    classes = classes[:n]
    scores = scores[:n]

    return boxes_det, scores, classes


def get_labels(label_map_path):
    categories = {}
    with open(label_map_path) as fp:
        for line in fp:
            line = line.strip()
            if 'id:' in line:
                obj_id = int(line.split('id:')[1].strip())
            elif 'name' in line:
                obj_name = line.split('\'')[1] if len(line.split('\'')) == 3 else line.split('"')[1]
                categories[obj_id] = obj_name
    return categories