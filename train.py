import tensorflow as tf
import tensorflow_datasets as tfds
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Step 2: Prepare the dataset
def filter_cat_class(dataset):
    def filter_fn(x):
        return tf.reduce_any(tf.equal(x['objects']['label'], 17))  # 17 is the label for cat in COCO
    return dataset.filter(filter_fn)

# Load COCO dataset
(train_ds, val_ds), ds_info = tfds.load('coco/2017', split=['train', 'validation'], with_info=True, as_supervised=False)

# Filter dataset to include only cat class
train_ds = filter_cat_class(train_ds)
val_ds = filter_cat_class(val_ds)

# Preprocess the dataset
def preprocess(data):
    image = tf.image.resize(data['image'], (320, 320))
    label = data['objects']
    return image, label

train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Step 3: Define the model
model_name = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
pipeline_config = f'{model_name}/pipeline.config'
model_dir = model_name

configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=True)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(f'{model_name}/checkpoint/ckpt-0').expect_partial()

# Step 4: Train the model with quantization-aware training
@tf.function
def train_step(image_tensors, groundtruth_boxes_list, groundtruth_classes_list):
    with tf.GradientTape() as tape:
        preprocessed_images = tf.concat([detection_model.preprocess(image_tensor)[0] for image_tensor in image_tensors], axis=0)
        prediction_dict = detection_model.predict(preprocessed_images, shapes)
        losses_dict = detection_model.loss(prediction_dict, groundtruth_boxes_list, groundtruth_classes_list)
        total_loss = losses_dict['Loss/total_loss']
    gradients = tape.gradient(total_loss, detection_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, detection_model.trainable_variables))
    return total_loss

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_ds:
        groundtruth_boxes_list = [label['bbox'] for label in labels]
        groundtruth_classes_list = [label['label'] for label in labels]
        total_loss = train_step(images, groundtruth_boxes_list, groundtruth_classes_list)
        print(f'Epoch {epoch}, Loss: {total_loss.numpy()}')

# Step 5: Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open('cat_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)
