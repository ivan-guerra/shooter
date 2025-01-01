//! Object detection functionality using YOLO neural networks.
//!
//! This module provides implementations for human detection using YOLO (You Only Look Once)
//! deep learning models through OpenCV's DNN module. It includes:
//!
//! - `YoloConfig`: Configuration parameters for YOLO detection
//! - `DarknetModel`: A wrapper around OpenCV's DNN implementation for Darknet models
use crate::config::Yolo;
use opencv::{
    core::{Rect, Scalar, Size, Vector, CV_32F},
    dnn::{self},
    prelude::*,
};

/// Represents a Darknet model for object detection
///
/// This struct encapsulates a DNN (Deep Neural Network) model loaded from Darknet format
pub struct DarknetModel {
    net: dnn::Net,
    yolo_conf: Yolo,
}

impl DarknetModel {
    /// Creates a new DarknetModel instance from model configuration and weights files
    ///
    /// # Arguments
    ///
    /// * `model_cfg` - Path to the Darknet model configuration file
    /// * `model_weights` - Path to the Darknet model weights file
    ///
    /// # Returns
    ///
    /// * `Result<Self, opencv::Error>` - A new DarknetModel instance or an OpenCV error
    pub fn new(yolo_conf: &Yolo) -> Result<Self, opencv::Error> {
        let mut net = dnn::read_net_from_darknet(
            yolo_conf
                .model_cfg
                .to_str()
                .expect("Invalid model config path"),
            yolo_conf
                .model_weights
                .to_str()
                .expect("Invalid model weights path"),
        )?;
        net.set_preferable_backend(dnn::DNN_BACKEND_DEFAULT)?;
        net.set_preferable_target(dnn::DNN_TARGET_CPU)?;

        Ok(Self {
            net,
            yolo_conf: yolo_conf.clone(),
        })
    }

    /// Detects humans in the provided image using a YOLO neural network.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image as OpenCV Mat
    ///
    /// # Returns
    ///
    /// * `opencv::Result<Vec<opencv::core::Rect>>` - Vector of bounding boxes around detected humans
    pub fn find_humans(
        &mut self,
        image: &opencv::core::Mat,
    ) -> opencv::Result<Vec<opencv::core::Rect>> {
        let (height, width) = (image.rows() as f32, image.cols() as f32);
        let input_blob = dnn::blob_from_image(
            &image,
            self.yolo_conf.scale_factor,
            Size::new(self.yolo_conf.input_size, self.yolo_conf.input_size),
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

    /// Processes the neural network output to extract human detections
    ///
    /// # Arguments
    ///
    /// * `width` - Original image width
    /// * `height` - Original image height
    ///
    /// # Returns
    ///
    /// * `opencv::Result<Vec<(Rect, f32, i32)>>` - Vector of tuples containing:
    ///   - Bounding box rectangle
    ///   - Confidence score
    ///   - Class ID (0 for person)
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

                if confidence > self.yolo_conf.confidence_threshold {
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

    /// Calculates the bounding box coordinates from YOLO detection data
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of detection data containing normalized coordinates
    /// * `width` - Original image width
    /// * `height` - Original image height
    ///
    /// # Returns
    ///
    /// * `Rect` - OpenCV rectangle representing the bounding box with coordinates
    ///           adjusted to the original image dimensions
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

    /// Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
    ///
    /// # Arguments
    ///
    /// * `boxes` - Vector of bounding box rectangles
    /// * `confidences` - Vector of confidence scores corresponding to each box
    ///
    /// # Returns
    ///
    /// * `opencv::Result<Vec<Rect>>` - Filtered vector of bounding boxes after NMS
    ///                                 or an OpenCV error
    fn apply_nms(&self, boxes: Vec<Rect>, confidences: Vec<f32>) -> opencv::Result<Vec<Rect>> {
        let mut indices = Vector::new();
        dnn::nms_boxes(
            &Vector::from(boxes.clone()),
            &Vector::from(confidences),
            self.yolo_conf.nms_confidence_threshold,
            self.yolo_conf.nms_threshold,
            &mut indices,
            self.yolo_conf.score_threshold,
            self.yolo_conf.top_k,
        )?;

        Ok(indices.iter().map(|idx| boxes[idx as usize]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Helper function to create a test model instance
    fn create_test_model() -> DarknetModel {
        let yolo_conf = Yolo::default();
        DarknetModel::new(&yolo_conf).unwrap()
    }

    #[test]
    fn darknetmodel_new_valid_paths() {
        let yolo_conf = Yolo::default();
        let result = DarknetModel::new(&yolo_conf);
        assert!(result.is_ok());
    }

    #[test]
    fn darknetmodel_new_invalid_paths() {
        let yolo_conf = Yolo {
            model_cfg: PathBuf::from("nonexistent.cfg"),
            model_weights: PathBuf::from("nonexistent.weights"),
            ..Default::default()
        };

        let result = DarknetModel::new(&yolo_conf);
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
