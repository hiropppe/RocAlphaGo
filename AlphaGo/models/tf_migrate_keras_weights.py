import glob
import h5py as h5
import os
import re
import tensorflow as tf

from AlphaGo.models.tf_policy import CNNPolicy


flags = tf.app.flags
flags.DEFINE_string('keras_weights', None, 'Keras policy weights file to migrate')
flags.DEFINE_string('keras_weights_directory', None, 'Directory which contains Keras policy weights files to migrate')
flags.DEFINE_string('output_directory', '/tmp/logs/sl_policy/', 'Output directory')
flags.DEFINE_integer('filters', 192, 'Number of filters')
FLAGS = flags.FLAGS


def main(argv=None):
    if FLAGS.keras_weights:
        migrate(FLAGS.keras_weights, FLAGS.output_directory)
    elif FLAGS.keras_weights_directory:
        for weights in glob.glob(os.path.join(FLAGS.keras_weights_directory, 'weights.*.hdf5')):
            epoch = re.findall(r'\d{5}', weights)[0]
            output_directory = os.path.join(FLAGS.output_directory, epoch)
            os.mkdir(output_directory)
            migrate(weights, output_directory)


def migrate(keras_weights, output_directory):
    print('Migrate Keras weights to TF checkpoint file. {}'.format(keras_weights))
    keras_model_weights = h5.File(keras_weights)['model_weights']

    def keras_weight(layer_i, wb, scope_name):
        layer = keras_model_weights[scope_name]
        if wb.upper() == 'W':
            if scope_name + '_W:0' in layer:
                value = layer[scope_name + '_W:0'].value
            else:
                value = layer[scope_name + '_W'].value

            value = value.transpose((2, 3, 1, 0))
            # Transpose kernel dimention ordering.
            # TF uses the last dimension as channel dimension,
            # K kernel shape: (output_depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, output_depth)
            return tf.Variable(value, name=scope_name + '_W')
        elif wb.lower() == 'b':
            if layer_i == 14:
                if 'Variable:0' in layer:
                    return tf.Variable(layer['Variable:0'].value, name='Variable')
                else:
                    return tf.Variable(layer['param_0'].value, name='Variable')
            else:
                if scope_name + '_b:0' in layer:
                    return tf.Variable(layer[scope_name + '_b:0'].value, name=scope_name + '_b')
                else:
                    return tf.Variable(layer[scope_name + '_b'].value, name=scope_name + '_b')

    policy = CNNPolicy(checkpoint_dir=output_directory,
                       filters=FLAGS.filters)
    policy.init_graph(weight_setter=keras_weight)
    policy.start_session()

    policy.save_model(global_step=0)


if __name__ == '__main__':
    tf.app.run(main=main)
