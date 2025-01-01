# shooter

A Nerf turret gun that automatically detects, tracks, and fires at humans.
**This is a work in progress**. Currently, we are able to render a live video
feed demonstrating the software can recognize a human. The software generates
azimuth and elevation angles to track their movement within the camera's field
of view. Below is a recording of `shooter` in action:

https://github.com/user-attachments/assets/5b448016-83db-4d57-be2f-94c9ee887f96

When a person is detected, a green bounding box is drawn around them. The red
dot represents the center of the bounding box and is where the turret gun will
aim. At the top of the screen, you can see live azimuth and elevation angles.
These angles will be converted to motor movements to aim the turret at the
person.

### Human Detection

A pretrained [You Only Look Once (YOLO)][1] model is used to detect humans.
`shooter` can be configured to use any YOLO model. Included in this project
under [`models/`](models/) is the [yolov4-tiny][2] model. We went with
yolo4-tiny because it's lightweight enough to run on a Raspberry Pi like the one
installed on our turret gun.

### Configuration

`shooter` requires a configuration file to run . You can checkout an example
config file under [`configs/test.toml`](configs/test.toml). The config file
contains sections for configuring camera settings and YOLO model parameters.

### Getting a Video Stream

`shooter` requires a video stream from a webcam or other video source. We
currently are using a [Nexigo N60 webcam][5] and the [cam2ip][3] software to
produce a video stream. You can [download][4] a cam2ip release binary from
GitHub. To produce an IP video feed, plugin your webcam and run:

```bash
cam2ip -index CAMERA_INDEX -height 416 -width 416 -delay 10
```

Where `CAMERA_INDEX` is the index of your webcam. You can use a utility like
`v4l2-ctl` to discover the index of your webcam. For example,

```text
v4l2-ctl --list-devices

NexiGo N60 FHD Webcam Audio: Ne (usb-0000:00:14.0-1.4):
	/dev/video2
	/dev/video3
	/dev/media1
```

In the output above, the first `/dev/video2` indicates that my webcam's index is 2.

Assuming you ran `cam2ip` as shown above, you can view your webcam's live video
stream in your browser using the URL `http://localhost:56000/mjpeg`. **You'll want
to make sure this URL is copied to the `stream_url` field in your config file**.

### Running `shooter`

Running `shooter` is a three step process:

1. Plugin your webcam.

2. Run `cam2ip` to produce a video stream (see [Getting a Video Stream](#getting-a-video-stream)).

3. Run `shooter` supplying it a configuration file:

```bash
shooter test.toml
```

[1]: https://github.com/AlexeyAB/darknet
[2]: https://github.com/AlexeyAB/darknet?tab=readme-ov-file#pre-trained-models
[3]: https://github.com/gen2brain/cam2ip
[4]: https://github.com/gen2brain/cam2ip/releases/tag/1.6
[5]: https://www.amazon.com/gp/product/B088TSR6YJ/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1
