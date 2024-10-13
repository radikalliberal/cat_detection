import tensorflow as tf
import tensorflow_datasets as tfds
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def create_tf_example(image, annotations, label_map_dict):
    height = image.shape[0]
    width = image.shape[1]
    filename = image['file_name'].encode('utf8')
    encoded_image_data = tf.io.encode_jpeg(image).numpy()
    image_format = b'jpeg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for annotation in annotations:
        xmins.append(annotation['bbox'][0] / width)
        xmaxs.append((annotation['bbox'][0] + annotation['bbox'][2]) / width)
        ymins.append(annotation['bbox'][1] / height)
        ymaxs.append((annotation['bbox'][1] + annotation['bbox'][3]) / height)
        classes_text.append(annotation['category_id'].encode('utf8'))
        classes.append(label_map_dict[annotation['category_id']])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def convert_to_tfrecord(dataset, output_path, label_map_dict):
    writer = tf.io.TFRecordWriter(output_path)
    for example in dataset:
        image = example['image']
        annotations = example['objects']
        tf_example = create_tf_example(image, annotations, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()

def main():
    # Load the COCO dataset
    train_dataset, info = tfds.load('coco/2017', split='train', with_info=True)
    val_dataset = tfds.load('coco/2017', split='validation')

    # Load the label map
    label_map_path = 'coco/mscoco_label_map.pbtxt'  # Update with the correct path
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)

    # Convert the datasets to TFRecord format
    convert_to_tfrecord(train_dataset, 'coco/train.record', label_map_dict)
    convert_to_tfrecord(val_dataset, 'coco/val.record', label_map_dict)

if __name__ == '__main__':
    main()

