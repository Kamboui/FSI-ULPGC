# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """

    o_h = np.zeros(n)
    o_h[x] = 1.
    return o_h


num_classes = 3
batch_size = 4


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.layers.flatten(o4), units=30, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["FSI/piedrafsi/train/*.jpg", "FSI/tecladofsi/train/*.jpg", "FSI/regletafsi/train/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["FSI/piedrafsi/valid/*.jpg", "FSI/tecladofsi/valid/*.jpg", "FSI/regletafsi/valid/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["FSI/piedrafsi/test/*.jpg", "FSI/tecladofsi/test/*.jpg", "FSI/regletafsi/test/*.jpg"], batch_size=56)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, dtype=tf.float32)))







# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0095).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    errorActual=0
    errorAnterior=0
    grafica1 = []
    grafica2 = []
    ajustado = 0
    epoca = 0
    while(ajustado < 3 or errorActual > 0.5 and epoca < 300):
        sess.run(optimizer)
        print("-----------------------------------")
        print ("Iteración: {} Estabilidad: {}".format(epoca, ajustado))

        errorEntrenamiento = sess.run(cost)
        errorActual = sess.run(cost_valid)

        grafica1.append(errorEntrenamiento)
        grafica2.append(errorActual)

        print("Error entrenamiento: {}, error validacion {}".format(errorEntrenamiento, errorActual))

        if (errorActual >= errorAnterior * 0.888):
            ajustado += 1;
        else:
            ajustado = 0

        #if _ % 20 == 0:
            #print("Iter:", _, "---------------------------------------------")
            #sess.run(label_batch_train)
            #sess.run(example_batch_train_predicted)
            #print("Error entrenamiento:", sess.run(cost))
            #sess.run(label_batch_valid)
            #sess.run(example_batch_valid_predicted)
            #print("Error validacion:", sess.run(cost_valid))
        errorAnterior = errorActual
        epoca +=1

    result_esperado = sess.run(label_batch_test)
    result = sess.run(example_batch_test_predicted)

    Aciertos = 0
    Fallos = 0

    for b, r in zip(result_esperado, result):
        if (np.argmax(b) == np.argmax(r)):
            Aciertos += 1
        else:
            Fallos += 1
    print("----------------------------------------------------------------------------------")
    print (result)
    print(result_esperado)
    print("----------------------------------------------------------------------------------")
    print("Aciertos: ", Aciertos)
    print("Fallos: ", Fallos)
    Total = Aciertos + Fallos
    print("Porcentaje de aciertos: ", (float(Aciertos) / float(Total)) * 100, "%")
    print("----------------------------------------------------------------------------------")
    #


    plt.plot(grafica1)
    plt.plot(grafica2)
    plt.legend(['Error Entrenamiento', 'Error Validacion'], loc='upper right')
    plt.show()
    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)