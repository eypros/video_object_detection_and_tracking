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
|  `-v`,<br>`--video-input-path` | The path to the video to be processed | `python3 object_detection_tracking.py -v="path-to-video"` | Required |
|  `--video-output-path` | Where to write the output video | `python3 object_detection_tracking.py --video-output-path="path-to-output-video"` | `(input_video_name)_detected(.ext)` |
| `-m`,<br>`--model-path` | Path to the frozen model to be used | `python3 object_detection_tracking.py --model-path="path-to-model/frozen_inference_graph.pb"` | `model/frozen_inference_graph.pb` |
| `-l`,<br>`--labels-map-path` | Path to the labels map to be used. | `python3 object_detection_tracking.py --labels-map-path="path-to-labels-map/labels.pbtxt"` | `model/label_map.pbtxt` | |
| `--detection-rate` | The rate of detection, it perform 1 detection every detection-rate value. |  `python3 object_detection_tracking.py --detection-rate=10 ` | `5` |

## How to use the project

The simpler way is to provide a path to a video to be processed using
the provided model.

`python3 object_detection_tracking.py --video-input-path="path-to-input-video/test_video.mp4"`

* this way a new output video will be created named:
`test_video_detected.mp4`. This can be changed using the arguments above.

* In this video all detected objects with confidence score >= 0.5 will be added a bounding box around them

* Object tracking will be perfomed every 5 frames for each of the above detected objects.

* After the tracking a new object detection will be performed