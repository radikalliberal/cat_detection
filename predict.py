
import tensorflow as tf
import numpy as np
import cv2
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util

# Load the trained model
model_name = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
pipeline_config = f'{model_name}/pipeline.config'
model_dir = model_name

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(f'{model_name}/checkpoint/ckpt-0').expect_partial()

# Load label map
label_map_path = f'{model_name}/label_map.pbtxt'  # Update with the correct path
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

def load_image_into_numpy_array(path):
    return np.array(cv2.imread(path))

def annotate_image(image, detections):
    image_np = cv2.resize(image, (None, None), fx=0.1, fy=0.1)
    side_max = max(image_np.shape[:2])
    h, w = image_np.shape[:2]


def preprocess_image(image):
    h, w, _ = image.shape
    if h >= w:
    image = cv2.resize(image, (320, int(320 * h / w)))
        h, w = image.shape[:2]
        cv2.copyMakeBorder(image, 0, 0, 0, h-w, cv2.BORDER_CONSTANT)
    else:
        image = cv2.resize(image, (int(320 * w / h), 320))
        h, w = image.shape[:2]
        cv2.copyMakeBorder(image, 0, w-h, 0, 0, cv2.BORDER_CONSTANT)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = tf.cast(input_tensor, tf.float32)
    return input_tensor

def detect(image_path):
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = preprocess_image(image_np)

    detections = detection_model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    image_np = cv2.resize(image_np, (None, None), fx=0.1, fy=0.1)
    side_max = max(image_np.shape[:2])
    h, w = image_np.shape[:2]

    detections['detection_boxes'] /= np.array([h / side_max, w / side_max, h / side_max, w / side_max])
    detections['detection_boxes'] *= np.array([h, w, h, w])
    # Visualization of the results of a detection.
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=False,
        line_thickness=max(1, int(side_max / 200)))

    cv2.imshow('object_detection', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        # print("Usage: python predict.py <path_to_image>")
        image_path = 'cat.jpg'
    else: 
        image_path = sys.argv[1]
    detect(image_path)
