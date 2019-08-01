import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import scipy


def get_filename_list(path, config):
    fd = open(path)
    image_filenames = []
    label_filenames = []
    for i in fd:
        i = i.strip().split(" ")
        image_filenames.append(i[0])
        label_filenames.append(i[1])

    image_filenames = [config["IMG_PREFIX"] + name for name in image_filenames]
    label_filenames = [config["LABEL_PREFIX"] + name for name in label_filenames]
    return image_filenames, label_filenames


def dataset_reader(filename_queue, config):

    image_filename = filename_queue[0]
    label_filename = filename_queue[1]

    imageValue = tf.read_file(image_filename)
    labelValue = tf.read_file(label_filename)

    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue)

    image = tf.reshape(image_bytes, (config["INPUT_HEIGHT"], config["INPUT_WIDTH"], config["INPUT_CHANNELS"]))
    label = tf.reshape(label_bytes, (config["INPUT_HEIGHT"], config["INPUT_WIDTH"], 1))

    return image, label


def dataset_inputs(image_filenames, label_filenames, batch_size, config):
    images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
    labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

    filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

    image, label = dataset_reader(filename_queue, config)
    reshaped_image = tf.cast(image, tf.float32)
    min_queue_examples = 300

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(reshaped_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 1
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('training_images', images)
    print('generating image and label batch:')
    return images, label_batch


def get_all_test_data(im_list, la_list):
    images = []
    labels = []
    index = 0
    for im_filename, la_filename in zip(im_list, la_list):
        im = scipy.misc.imread(im_filename)
        la = scipy.misc.imread(la_filename)
        images.append(im)
        labels.append(la)
        index = index + 1

    print('%d CamVid test images are loaded' % index)
    return images, labels