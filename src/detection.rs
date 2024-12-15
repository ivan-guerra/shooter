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
        let input_blob = dnn::blob_from_image(
            &image,
            YOLO_CONFIG.scale_factor,
            Size::new(YOLO_CONFIG.input_size, YOLO_CONFIG.input_size),
            Scalar::new(0.0, 0.0, 0.0, 0.0),
            true,
            false,
            CV_32F,
        )?;

        self.net
            .set_input(&input_blob, "", 1.0, Scalar::default())?;

        let detections = self.process_network_output(width, height)?;
        let (boxes, confidences): (Vec<_>, Vec<_>) = detections
            .into_iter()
            .map(|(rect, conf, _)| (rect, conf))
            .unzip();

        self.apply_nms(boxes, confidences)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Helper function to create a test model instance
    fn create_test_model() -> DarknetModel {
        let model_cfg = PathBuf::from("models/yolov4-tiny.cfg");
        let model_weights = PathBuf::from("models/yolov4-tiny.weights");
        DarknetModel::new(&model_cfg, &model_weights).unwrap()
    }

    #[test]
    fn darknetmodel_new_valid_paths() {
        let model_cfg = PathBuf::from("models/yolov4-tiny.cfg");
        let model_weights = PathBuf::from("models/yolov4-tiny.weights");

        let result = DarknetModel::new(&model_cfg, &model_weights);
        assert!(result.is_ok());
    }

    #[test]
    fn darknetmodel_new_invalid_paths() {
        let model_cfg = PathBuf::from("nonexistent.cfg");
        let model_weights = PathBuf::from("nonexistent.weights");

        let result = DarknetModel::new(&model_cfg, &model_weights);
        assert!(result.is_err());
    }

    mod calculate_bbox_tests {
        use super::*;

        #[test]
        fn center_box() {
            let model = create_test_model();
            let data = vec![0.5, 0.5, 0.2, 0.2];
            let (width, height) = (100.0, 100.0);

            let bbox = model.calculate_bbox(&data, width, height);

            assert_eq!(bbox.x, 40);
            assert_eq!(bbox.y, 40);
            assert_eq!(bbox.width, 20);
            assert_eq!(bbox.height, 20);
        }

        #[test]
        fn corner_box() {
            let model = create_test_model();
            let data = vec![0.1, 0.1, 0.2, 0.2];
            let (width, height) = (100.0, 100.0);

            let bbox = model.calculate_bbox(&data, width, height);

            assert_eq!(bbox.x, 0);
            assert_eq!(bbox.y, 0);
            assert_eq!(bbox.width, 20);
            assert_eq!(bbox.height, 20);
        }

        #[test]
        fn edge_box() {
            let model = create_test_model();
            let data = vec![0.9, 0.9, 0.2, 0.2];
            let (width, height) = (100.0, 100.0);

            let bbox = model.calculate_bbox(&data, width, height);

            assert_eq!(bbox.x, 80);
            assert_eq!(bbox.y, 80);
            assert_eq!(bbox.width, 20);
            assert_eq!(bbox.height, 20);
        }
    }

    mod apply_nms_tests {
        use super::*;

        #[test]
        fn no_overlapping_boxes() {
            let model = create_test_model();
            let boxes = vec![
                Rect::new(0, 0, 10, 10),
                Rect::new(20, 20, 10, 10),
                Rect::new(40, 40, 10, 10),
            ];
            let confidences = vec![0.9, 0.8, 0.7];

            let result = model.apply_nms(boxes.clone(), confidences).unwrap();

            assert_eq!(result.len(), 3);
            assert!(result.contains(&boxes[0]));
            assert!(result.contains(&boxes[1]));
            assert!(result.contains(&boxes[2]));
        }

        #[test]
        fn overlapping_boxes() {
            let model = create_test_model();
            let boxes = vec![
                Rect::new(0, 0, 20, 20),
                Rect::new(19, 19, 20, 20),
                Rect::new(40, 40, 20, 20),
            ];
            let confidences = vec![0.9, 0.7, 0.8];

            let result = model.apply_nms(boxes.clone(), confidences).unwrap();

            assert_eq!(result.len(), 3);
            assert!(result.contains(&boxes[0]));
            assert!(result.contains(&boxes[1]));
            assert!(result.contains(&boxes[2]));
        }

        #[test]
        fn low_confidence() {
            let model = create_test_model();
            let boxes = vec![Rect::new(0, 0, 10, 10), Rect::new(20, 20, 10, 10)];
            let confidences = vec![0.3, 0.2];

            let result = model.apply_nms(boxes, confidences).unwrap();

            assert_eq!(result.len(), 0);
        }

        #[test]
        fn empty_input() {
            let model = create_test_model();
            let result = model.apply_nms(vec![], vec![]).unwrap();
            assert_eq!(result.len(), 0);
        }

        #[test]
        fn single_box() {
            let model = create_test_model();
            let boxes = vec![Rect::new(0, 0, 10, 10)];
            let confidences = vec![0.9];

            let result = model.apply_nms(boxes.clone(), confidences).unwrap();

            assert_eq!(result.len(), 1);
            assert_eq!(result[0], boxes[0]);
        }
    }
}
