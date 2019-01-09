import tensorflow as tf

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map()  # Parse the record into tensors.
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
