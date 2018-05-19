""""import gzip
import pickle as cPickle
import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """"""
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """"""
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set


# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()  # Let's see a sample
print (train_y[57])


# TODO: the neural net!!
"""""
import gzip
import pickle as cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set


valid_x, valid_y = valid_set


test_x, test_y = test_set
# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.pyplot as plt

# TODO: the neural net!!

# Las etiquetas están en la última fila. Las codificamos con el one_hot
train_y = one_hot(train_y, 10)
valid_y = one_hot(valid_y, 10)
test_y = one_hot(test_y, 10)


# Placeholder, esto es un tipo de variable que estará constantemente cambiando
# Es decir que en un inicio esta vacía, pero a medida que avanzamos la vamos modificando
# Normalmente se usan para los inputs
# Matriz de entrada
x = tf.placeholder(tf.float32, [None, 784])    # IMAGEN DEL NUMERO(MNIST) DESCOMPUESTA EN UN VWCTOR

# labels = etiqueta [Se pone 10 ya que representa el numero que es cada muestra en forma de vector(one_hot)]
# Matriz con las etiquetas REALES del set de datos mnist
y_ = tf.placeholder(tf.float32, [None, 10]) # MATRIZ CON ETIQUETAS REALES DEL SET DE DATOS

# ESTA ES LA CAPA OCULTA (INPUT)
W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)   # MATRIZ DE PESOS, 784 PARA RECIBIR LA IMAGEN, 10 PARA POSIBLES SALIDA
b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)      # VECTOR CON BIAS

# ESTA ES LA SALIDA DE LAS NEURONAS (OUTPUT)
W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)     # MATRIZ DE PESOS, 10 PARA RECIBIR LA IMAGEN, 10 PARA POSIBLES SALIDA
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)      # VECTOR CON BIAS

#   Prueba 1
# W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)       ||  Learning Rate: 0,01
# b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)            ||  Resultados: batch: 20
# W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)        ||              Épocas: 31
# b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)            ||              Porcentaje de acierto: 90,3%

#   Prueba 2
# W1 = tf.Variable(np.float32(np.random.rand(784, 15)) * 0.1)       ||  Learning Rate: 0,01
# b1 = tf.Variable(np.float32(np.random.rand(15)) * 0.1)            ||  Resultados: batch: 20
# W2 = tf.Variable(np.float32(np.random.rand(15, 10)) * 0.1)        ||              Épocas: 20
# b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)            ||              Porcentaje de acierto: 91,8%

#   Prueba 3
# W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)       ||  Learning Rate: 0,01
# b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)            ||  Resultados: batch: 25
# W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)        ||              Épocas: 27
# b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)            ||              Porcentaje de acierto: 90.5%

#   Prueba 4
# W1 = tf.Variable(np.float32(np.random.rand(784, 15)) * 0.1)       ||  Learning Rate: 0,01
# b1 = tf.Variable(np.float32(np.random.rand(15)) * 0.1)            ||  Resultados: batch: 25
# W2 = tf.Variable(np.float32(np.random.rand(15, 10)) * 0.1)        ||              Épocas: 29
# b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)||          ||              Porcentaje de acierto: 91.9%

#   Prueba 5
# W1 = tf.Variable(np.float32(np.random.rand(784, 20)) * 0.1)       ||  Learning Rate: 0,03
# b1 = tf.Variable(np.float32(np.random.rand(20)) * 0.1)            ||  Resultados: batch: 25
# W2 = tf.Variable(np.float32(np.random.rand(20, 10)) * 0.1)        ||              Épocas: 22
# b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)            ||              Porcentaje de acierto: 92.9%

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)    # Capa de salida
y = tf.nn.sigmoid(tf.matmul(h, W2) + b2)    # Salida

# -------------------------------------------------------------------------------------
loss = tf.reduce_sum(tf.square(y_ - y))
train = tf.train.GradientDescentOptimizer(0.03).minimize(loss)  # learning rate: 0.01
# -------------------------------------------------------------------------------------


# Este array se encarga de decirlo que numeros clasifico bien y cuales mal
# Basicamente compara el resultado obtenido contra el resultado teorico, si esta bien GG si no GET REKT
prediccion = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# Nos devuelve un porcentaje(reduce_mean) de certeza
accuracy = tf.reduce_mean(tf.cast(prediccion, tf.float32))



# Preguntar por el warning y cambie el tf.initialize_all_variables() -> tf.global_variables_initializer()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


print ("----------------------")
print ("   Start training...  ")
print ("----------------------")


errorPrevioV = 0;
errorPrevioE = 0;
actualLossValid = 0;
batch_size = 25
graf_error = []
graf_error_v = []
epoch = 0
ajustado = 0;

while (ajustado < 15):
    for jj in range(int(len(train_x) / batch_size)):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    actualLossTrain = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})/batch_size
    actualLossValid = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})/len(valid_y)

    if( actualLossValid >= errorPrevioV * 0.95 and actualLossTrain >= errorPrevioE * 0.95):
        ajustado += 1;
    else:
        ajustado = 0

    graf_error.append(actualLossTrain)
    graf_error_v.append(actualLossValid)

    print("Epoca {} error {} error V {}  ajuste{}".format(epoch, actualLossTrain, actualLossValid, ajustado))

#
    #result = sess.run(y, feed_dict={x: batch_xs})
#
    #for b, r in zip(batch_ys, result):
    #    print(b, "--> ", r)
    #print("----------------------------------------------------------------------------------")
    errorPrevioV = actualLossValid;
    errorPrevioE = actualLossTrain;
    epoch = epoch + 1
porcentaje = sess.run(accuracy, feed_dict={x: test_x, y_: test_y}) * 100
print("Porcentaje de aciertos en el test: {:.3}%".format(porcentaje))

# Para añadir otra grafica encima de otra basta con poner en la misma funcion plot, otra lista al final
#x_axis_lista_certeza = list(range(1, len(lista_certeza) + 1))
#plt.plot(x_axis_lista_certeza, lista_certeza)
#plt.show()


plt.plot(graf_error)
plt.plot(graf_error_v)
plt.legend(['Error Entrenamiento', 'Error Validacion'], loc='upper right')
plt.show()