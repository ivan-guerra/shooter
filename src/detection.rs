use opencv::{
    core::{Rect, Scalar, Size, Vector, CV_32F},
    dnn::{self},
    imgcodecs, imgproc,
    prelude::*,
};

pub struct DarknetModel {
    net: dnn::Net,
}

const YOLO_CONFIG: YoloConfig = YoloConfig {
    input_size: 416,
    scale_factor: 1.0 / 255.0,
    confidence_threshold: 0.5,
    nms_confidence_threshold: 0.5,
    nms_threshold: 0.4,
    score_threshold: 1.0,
    top_k: 0,
};

struct YoloConfig {
    input_size: i32,
    scale_factor: f64,
    confidence_threshold: f32,
    nms_confidence_threshold: f32,
    nms_threshold: f32,
    score_threshold: f32,
    top_k: i32,
}

impl DarknetModel {
    pub fn new(
        model_cfg: &std::path::Path,
        model_weights: &std::path::Path,
    ) -> Result<Self, opencv::Error> {
        let mut net = dnn::read_net_from_darknet(
            model_cfg.to_str().expect("Invalid model config path"),
            model_weights.to_str().expect("Invalid model weights path"),
        )?;
        net.set_preferable_backend(dnn::DNN_BACKEND_DEFAULT)?;
        net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

        Ok(Self { net })
    }

    pub fn find_humans(
        &mut self,
        image: &std::path::Path,
    ) -> opencv::Result<Vec<opencv::core::Rect>> {
        let image = imgcodecs::imread(
            image.to_str().expect("Invalid image path"),
            imgcodecs::IMREAD_COLOR,
        )?;
        let (height, width) = (image.rows() as f32, image.cols() as f32);

        let input_blob = self.prepare_input_blob(&image)?;
        self.net
            .set_input(&input_blob, "", 1.0, Scalar::default())?;

        let detections = self.process_network_output(width, height)?;
        let (boxes, confidences): (Vec<_>, Vec<_>) = detections
            .into_iter()
            .map(|(rect, conf, _)| (rect, conf))
            .unzip();

        self.apply_nms(boxes, confidences)
    }

    fn prepare_input_blob(&self, image: &Mat) -> opencv::Result<Mat> {
        dnn::blob_from_image(
            image,
            YOLO_CONFIG.scale_factor,
            Size::new(YOLO_CONFIG.input_size, YOLO_CONFIG.input_size),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,
            false,
            CV_32F,
        )
    }

    fn process_network_output(
        &mut self,
        width: f32,
        height: f32,
    ) -> opencv::Result<Vec<(Rect, f32, i32)>> {
        let mut outputs: Vector<Mat> = Vector::new();
        self.net
            .forward(&mut outputs, &self.net.get_unconnected_out_layers_names()?)?;

        let mut detections = Vec::new();

        for output in outputs {
            let data = output.data_typed::<f32>()?;
            let cols = output.cols() as usize;

            for row in 0..output.rows() as usize {
                let offset = row * cols;
                let confidence_range = &data[offset + 5..offset + cols];
                let confidence = *confidence_range
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(&0.0);

                if confidence > YOLO_CONFIG.confidence_threshold {
                    let class_id = confidence_range
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx as i32)
                        .unwrap_or(0);

                    // class_id 0 corresponds to the 'person' class in the COCO dataset
                    if class_id == 0 {
                        let bbox = self.calculate_bbox(&data[offset..], width, height);
                        detections.push((bbox, confidence, class_id));
                    }
                }
            }
        }

        Ok(detections)
    }

    fn calculate_bbox(&self, data: &[f32], width: f32, height: f32) -> Rect {
        let center_x = data[0] * width;
        let center_y = data[1] * height;
        let box_width = data[2] * width;
        let box_height = data[3] * height;

        Rect::new(
            ((center_x - box_width / 2.0).max(0.0)) as i32,
            ((center_y - box_height / 2.0).max(0.0)) as i32,
            (box_width.min(width - (center_x - box_width / 2.0).max(0.0))) as i32,
            (box_height.min(height - (center_y - box_height / 2.0).max(0.0))) as i32,
        )
    }

    fn apply_nms(&self, boxes: Vec<Rect>, confidences: Vec<f32>) -> opencv::Result<Vec<Rect>> {
        let mut indices = Vector::new();
        dnn::nms_boxes(
            &Vector::from(boxes.clone()),
            &Vector::from(confidences),
            YOLO_CONFIG.nms_confidence_threshold,
            YOLO_CONFIG.nms_threshold,
            &mut indices,
            YOLO_CONFIG.score_threshold,
            YOLO_CONFIG.top_k,
        )?;

        Ok(indices.iter().map(|idx| boxes[idx as usize]).collect())
    }

    pub fn draw_bounding_boxes(
        &self,
        input_image: &std::path::Path,
        output_image: &std::path::Path,
        boxes: &[opencv::core::Rect],
    ) -> opencv::Result<()> {
        let mut image = imgcodecs::imread(
            input_image.to_str().expect("Invalid input image path"),
            imgcodecs::IMREAD_COLOR,
        )?;

        for bbox in boxes {
            imgproc::rectangle(
                &mut image,
                *bbox,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }

        imgcodecs::imwrite(
            output_image.to_str().expect("Invalid output image path"),
            &image,
            &Vector::new(),
        )?;
        Ok(())
    }
}
