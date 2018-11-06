import load_data
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import model
import keras.backend as K
import numpy as np
import time



with tf.Graph().as_default():
    # Initialize the global step
    global_step = tf.Variable(0, trainable=False)
    # steps per epoch = total # of samples / batch size
    steps_per_epoch = np.ceil(200).astype(np.int32)
    # number of total steps = number of epoch * steps per epoch
    num_total_steps = 1000 * steps_per_epoch

    start_learning_rate = 1e-5

    boundaries = [np.int32((1 / 5) * num_total_steps), np.int32((2 / 5) * num_total_steps)]
    values = [1e-4, 1e-4, 1e-5]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

    left_path = '/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/a'
    right_path = '/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/b'

    train_image = load_data.load_data('train').image_loader(left_path, right_path)
    opt_step = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Initialize the model and put left and right images
    Model = model.model(mode='train', left=train_image[0], right=train_image[1])
    loss = Model.total_loss
    reuse_variables = True
    total_loss = K.mean(loss)
    # opt_step.minimize(loss=total_loss, global_step=global_step)

    # Calculate the gradients of every variables
    grads = opt_step.compute_gradients(total_loss)
    apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)


    tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
    tf.summary.scalar('total_loss', total_loss, ['model_0'])
    summary_op = tf.summary.merge_all('model_0')

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    summary_writer = tf.summary.FileWriter('/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/he', sess.graph)
    train_saver = tf.train.Saver()

    # COUNT PARAMS
    total_num_parameters = 0
    for variable in tf.trainable_variables():
        total_num_parameters += np.array(variable.get_shape().as_list()).prod()
    print("number of trainable parameters: {}".format(total_num_parameters))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # Start to train the network
    start_step = global_step.eval(session=sess)
    start_time = time.time()
    for step in range(start_step, num_total_steps):
        before_op_time = time.time()
        _, loss_value = sess.run([apply_gradient_op, total_loss])
        print(loss_value)
        duration = time.time() - before_op_time
        if step and step % 100 == 0:
            # examples_per_sec = 1 / duration
            time_sofar = (time.time() - start_time) / 3600
            training_time_left = (num_total_steps / step - 1.0) * time_sofar
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, global_step=step)
        if step and step % 10000 == 0:
            train_saver.save(sess, '/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/' + '/model', global_step=step)

    train_saver.save(sess, '/media/xiangtao/data/KITTI/data_scene_flow/training/dummy_training/' + '/model',
                     global_step=num_total_steps)

