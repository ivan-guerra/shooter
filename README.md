# shooter

A Nerf turret gun that automatically detects, tracks, and fires at humans.
**This is a work in progress**. Currently, we're able to detect humans and
generate commands for aiming the gun at a rate of 5-10 FPS.

`shooter` is split into a client and server. The client, `tgc` (turret gun
client), runs onboard the Rapsberry Pi. The client requests the latest command
from the server and then executes it by moving the motors/firing the gun. The
server, `tgs` (turret gun server), runs on a more powerful machine and processes
the image stream generating commands that are sent to the client only upon
request. The client/server architecture lets us keep the software onboard the Pi
lean while giving us the ability to generate detections at a high rate on a
separate, more powerful machine.

### Human Detection

A pretrained [You Only Look Once (YOLO)][1] model is used to detect humans.
`shooter` can be configured to use any YOLO model. Included in this project
under [`models/`](models/) is the [yolov4-tiny][2] model. We went with
yolo4-tiny because it's accurate enough to meet our goal and we're able to
perform inference at 5-10 FPS on modest hardware.

### Configuration

`shooter` requires a configuration file to run. You can checkout an example
config file under [`configs/test.toml`](configs/test.toml).

### Getting a Video Stream

`shooter` requires a video stream from a webcam or other video source. We
currently are using a [Nexigo N60 webcam][5] and the [cam2ip][3] software to
produce a video stream. You can [download][4] a cam2ip release binary from
GitHub.

### Running `tgc`

1. Cross compile the `tgc` binary for the Raspberry Pi. See
   [`docker/README.md`](docker/README.md) for instructions.

2. Copy the `tgc` binary, configs, and models to the Raspberry Pi:

```bash
scp target/aarch64-unknown-linux-gnu/release/tgc shooter@rpi:/home/shooter
scp -r configs/ shooter@rpi:/home/shooter
scp -r models/ shooter@rpi:/home/shooter
```

3. Plug the webcam into the Raspberry Pi.

4. Run `cam2ip` to produce a video stream.

5. Edit the `client` section of the configuration file as needed.

6. Run `tgc` supplying it a configuration file:

```bash
tgc configs/test.toml
```

### Running `tgs`

1. Install the following dependencies:

```bash
sudo apt-get install -y build-essential clang libclang-dev libopencv-dev
```

2. Build the `tgs` binary:

```bash
cargo build --release --bin tgs
```

3. Edit the `server` section of the configuration file as needed.

4. Run `tgs` supplying it a configuration file:

```bash
tgs configs/test.toml
```

[1]: https://github.com/AlexeyAB/darknet
[2]: https://github.com/AlexeyAB/darknet?tab=readme-ov-file#pre-trained-models
[3]: https://github.com/gen2brain/cam2ip
[4]: https://github.com/gen2brain/cam2ip/releases/tag/1.6
[5]: https://www.amazon.com/gp/product/B088TSR6YJ/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1
