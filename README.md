# shooter

A Nerf turret gun that automatically detects, tracks, and fires at humans.
**This is a work in progress**. Currently, we are able to render a live video
feed demonstrating the software can recognize a human. The software generates
azimuth and elevation angles to track their movement within the camera's field
of view. Below is a recording of the turret gun software in action:

https://github.com/user-attachments/assets/5b448016-83db-4d57-be2f-94c9ee887f96

When a person is detected, a green bounding box is drawn around them. The red
dot represents the center of the bounding box and is where the turret gun will
aim. At the top of the screen, you can see live azimuth and elevation angles.
These angles will be converted to motor movements to aim the turret at the
person.

The `shooter` project includes two applications. The first application is called
`tgun` (turret gun). `tgun` runs onboard the Raspberry Pi and is responsible for
controlling the turret gun. The second application is called `tlm` (telemetry)
and is mainly useful for debugging. `tlm` runs on a separate computer and reads
telemetry data sent by `tgun` over the network. `tlm` renders a live video feed
with overlays like the one shown in the demo video above.

### Human Detection

A pretrained [You Only Look Once (YOLO)][1] model is used to detect humans.
`shooter` can be configured to use any YOLO model. Included in this project
under [`models/`](models/) is the [yolov4-tiny][2] model. We went with
yolo4-tiny because it's lightweight enough to run on a Raspberry Pi like the one
installed on our turret gun.

### Configuration

`shooter` requires a configuration file to run . You can checkout an example
config file under [`configs/test.toml`](configs/test.toml). The config file
contains sections for configuring camera, detection model, and telemetry
parameters.

### Getting a Video Stream

`shooter` requires a video stream from a webcam or other video source. We
currently are using a [Nexigo N60 webcam][5] and the [cam2ip][3] software to
produce a video stream. You can [download][4] a cam2ip release binary from
GitHub.

### Running `tgun`

1. Cross compile the `tgun` binary for the Raspberry Pi. See [`docker/README.md`](docker/README.md) for instructions.

2. Copy the `tgun` binary, configs, and models to the Raspberry Pi:

```bash
scp target/aarch64-unknown-linux-gnu/release/tgun ieg@10.0.0.247:/home/ieg
scp -r configs/ ieg@10.0.0.247:/home/ieg
scp -r models/ ieg@10.0.0.247:/home/ieg
```

3. Plug the webcam into the Raspberry Pi.

4. Run `cam2ip` to produce a video stream.

5. Edit the configuration file and adjust parameters to match your network and
   camera.

6. Run `tgun` supplying it a configuration file:

```bash
tgun test.toml
```

### Running `tlm`

1. Build the `tlm` binary:

```bash
cargo build --release --bin tlm
```

2. Edit the configuration file as necessary. Note, usually the same
   configuration file given to `tgun` can be used with `tlm` without
   modification.

3. Run `tlm` supplying it a configuration file and the width/height of the
   video stream frames:

```bash
tlm --width 640 --height 480 test.toml
```

[1]: https://github.com/AlexeyAB/darknet
[2]: https://github.com/AlexeyAB/darknet?tab=readme-ov-file#pre-trained-models
[3]: https://github.com/gen2brain/cam2ip
[4]: https://github.com/gen2brain/cam2ip/releases/tag/1.6
[5]: https://www.amazon.com/gp/product/B088TSR6YJ/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1
