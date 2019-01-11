import tensorflow as tf

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
map_fun = lambda example: tf.parse_single_example(
                    example,
                    features={
                        'image/encoded': tf.FixedLenFeature([], tf.string),
                        'image/filename': tf.FixedLenFeature([], tf.string),
                        'image/ID': tf.FixedLenFeature([], tf.string),
                        'image/format': tf.FixedLenFeature([], tf.string),
                        'image/height': tf.FixedLenFeature([], tf.int64),
                        'image/width': tf.FixedLenFeature([], tf.int64),
                        'image/channels': tf.FixedLenFeature([], tf.int64),
                        'image/colorspace': tf.FixedLenFeature([], tf.string),
                        'image/segmentation/class/encoded': tf.FixedLenFeature([], tf.string),
                        'image/segmentation/class/format': tf.FixedLenFeature([], tf.string),
                        })
dataset = dataset.map(map_fun)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
training_filenames = ["/data/labels/person_attribute/export/train.record-00000-of-00010",
                      "/data/labels/person_attribute/export/train.record-00001-of-00010"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# Initialize `iterator` with validation data.
validation_filenames = ["/data/labels/person_attribute/export/val.record-00000-of-00010",
                        "/data/labels/person_attribute/export/train.record-00001-of-00010"]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
