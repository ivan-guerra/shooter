use clap::Parser;

pub mod detection;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(help = "input image file")]
    input_img: std::path::PathBuf,

    #[arg(help = "output image file")]
    output_img: std::path::PathBuf,
}

fn run(input_img: std::path::PathBuf, output_img: std::path::PathBuf) -> opencv::Result<()> {
    let mut model = detection::DarknetModel::new(
        std::path::Path::new("models/yolov4-tiny.cfg"),
        std::path::Path::new("models/yolov4-tiny.weights"),
    )?;
    let boxes = model.find_humans(&input_img)?;

    model.draw_bounding_boxes(&input_img, &output_img, &boxes)?;

    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(args.input_img, args.output_img) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
