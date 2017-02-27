import h5py as h5
import tensorflow as tf

from AlphaGo.models.tf_policy import CNNPolicy


flags = tf.app.flags
flags.DEFINE_string('keras_weights', None, 'Keras policy model file to migrate')
flags.DEFINE_string('output_directory', './logs', 'Output directory')
FLAGS = flags.FLAGS


def main(argv=None):
    keras_model_weights = h5.File(FLAGS.keras_weights)['model_weights']

    def keras_weight(layer_i, wb, scope_name):
        layer = keras_model_weights[scope_name]
        if wb.upper() == 'W':
            value = layer[scope_name + '_W:0'].value
            value = value.transpose((2, 3, 1, 0))
            # Transpose kernel dimention ordering.
            # TF uses the last dimension as channel dimension,
            # K kernel shape: (output_depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, output_depth)
            return tf.Variable(value, name=scope_name + '_W')
        elif wb.lower() == 'b':
            if layer_i == 14:
                return tf.Variable(layer['Variable:0'].value, name='Variable')
            else:
                return tf.Variable(layer[scope_name + '_b:0'].value, name=scope_name + '_b')

    policy = CNNPolicy(checkpoint_dir=FLAGS.output_directory)
    policy.init_graph(weight_setter=keras_weight)
    policy.start_session()

    policy.save_model(global_step=0)


if __name__ == '__main__':
    tf.app.run(main=main)
