# Cross Compilation Using Docker

`shooter` is meant to run on a Raspberry Pi meaning you have to cross compile
the project for the Pi. We cross compile using a docker image built from
`rpi-xcompile.Dockerfile`. `rpi-xcompile.Dockerfile` is a modified version of
the dockerfile provided by the [`opencv-rust`][1] project. The original file
targeted 32-bit arm. We modified it to target 64-bit arm since our Raspberry Pi
model 3B+ runs a 64-bit OS.

### Build Requirements

To cross compile the project, you must have the following packages installed:

- `docker`
- `qemu-arm`
- `binfmt-misc`

### Build Steps

To build the docker image, run the command that follows from the root of the
project. Set `ROOTFS` to point to the `root.tar.xz` of your Raspberry Pi's OS.
Visit https://downloads.raspberrypi.com/ to browse the options and be sure to
select a ARM64 rootfs matching that of your Pi.

```bash
docker build \
  -t rpi-xcompile \
  -f docker/rpi-xcompile.Dockerfile docker \
  --build-arg ROOTFS=https://downloads.raspberrypi.com/raspios_lite_arm64/archive/2024-11-19-15:18/root.tar.xz
```

The image will take a few minutes to build. Once built, you can run a container
to produce a release build of the project:

```bash
docker run --rm -v ./:/src rpi-xcompile
```

Once the build finishes, the binary will be located under
`target/aarch64-unknown-linux-gnu/release/[tgs|tgc]`.

[1]: https://github.com/twistedfall/opencv-rust/blob/master/tools/docker/rpi-xcompile.Dockerfile
