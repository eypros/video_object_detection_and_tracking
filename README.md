# Basic framework for object detection and tracking to videos

## What does this project has to offer

This is a basic framework for a functional object detection combined
with multiple object tracking.

The object detection is performed using tensorflow object detection api
while the tracking part is performed using opencv tracking.

## Arguments

| Argument &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description | Example | Default |
|:-------------:|:-----------:|:-----------:|:-----------:|
| `-h`,<br>`--help ` |	show help message | `python3 object_detection_tracking.py -h` | |
| `--version` | check version | `python3 object_detection_tracking.py -v` | |
| `--visualize` | Whether to visualize the results in every frame. | `python3 object_detection_tracking.py --visualize` | False |
| `--score-threshold` | The score above which bboxes are taken into consideration. | `python3 object_detection_tracking.py --score-threshold=0.7` | `0.5`|
|  `-v`,<br>`--video-path` | The path to the video to be processed | `python3 object_detection_tracking.py -v="path-to-video"` | Required |
| `-m`,<br>`--model-path` | Path to the frozen model to be used | `python3 object_detection_tracking.py --model-path="path-to-model/frozen_inference_graph.pb"` | `model/frozen_inference_graph.pb` |
| `-l`,<br>`--labels-map-path` | Path to the labels map to be used. | `python3 object_detection_tracking.py --labels-map-path="path-to-labels-map/labels.pbtxt"` | `model/label_map.pbtxt` | |
| `--detection-rate` | The rate of detection, it perform 1 detection every detection-rate value. |  `python3 object_detection_tracking.py --detection-rate=10 ` | `5` |


