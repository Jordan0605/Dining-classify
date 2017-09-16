import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
#from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
#from keras.models import Sequential

%matplotlib inline
plt.style.use('ggplot')

def read_data(file_path):
    col_names = ['timesyamp', 'x-axis', 'y-axis', 'z-axis', 'heart-rate', 'activity']
    data = pd.read_csv(file_path, header=None, names=col_names)
    return data

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y)-np.std(y), max(y)+np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def plot_activity(activity, data):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(15, 10), sharex=True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plot_axis(ax3, data['timestamp'], data['heart-rate'], 'heart-rate')
    plt.subplots_adjust(hspace=0.2)
    fig.subtitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def windows(data, size):
    start = 0
    while start < data.count():
        yield start, start+size
        start += (size/2)

def segment_signal(data,window_size=90):
    segments = np.empty((0, window_size, 4))
    labels = np.empty((0))
    for (start, end) in windows(data['timestamp'], window_size):
        x = data['x-axis'][start:end]
        y = data['y-axis'][start:end]
        z = data['z-axis'][start:end]
        hr = data['heart-rate'][start:end]
        if len(dataset['timestamp'][start:end]) == window_size:
            segments = np.vstack([segments, np.dstack([x, y, z, hr])])
            labels = np.append(labels, stats.mode(data['activity'][start:end])[0])
    return segments, labels

#cnn

def weight_variable(shape):
    initial = tf.truncates_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')

def apply_depthwise_conv(x, kernel_size, num_channels, depth):
    wights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights), biases))

def apply_max_pool(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1], strides=[1, 1, stride_size, 1], padding='VALID')

def main():
    dataset = read_data()
    dataset['x-axis'] = feature_noemalize(dataset['x-axis'])
    dataset['y-axis'] = feature_normalize(dataset['y-axis'])
    dataset['z-axis'] = feature_normalize(dataset['z-axis'])
    dataset['heart-rate'] = feature_normalize(dataset['heart-rate'])
    for activity in np.unique(dataset['activity']):
        subset = dataset[dataset['activity'] == activity][:180]
        plot_activity(activity, subset)

    segments, labels = segment_signal(dataset)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    reshaped_segments = segments.reshape(len(segments), 1, 90, 4)
    #70/30
    train_test_spilt = np.random.rand(len(reshaped_segments)) < 0.70
    train_x = reshaped_segments[train_test_spilt]
    train_y = labels[train_test_spilt]
    test_x = reshaped_segments[~train_test_spilt]
    test_y = labels[~train_test_spilt]

    total_batchs = train_x.shape[0]

    X = tf.placeholder(tf.float32, shape=[None, input_height, inputwidth, num_channels])
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])

    c = apply_depthwise_conv(X,kernel_size, num_channels, depth)
    p = apply_max_pool(c, 20, 2)
    c = apply_depthwise_conv(p, 6, depth*num_channels, depth/10)
    
    shape = c.get_shape().as_list()
    c_flat = tf.reshape(c, [-1, shape[1]*shape[2]*shape[3]])

    f_weights_l1 = weight_variable([shape[1]*shape[2]*depth*num_channels*(depth//10), num_hidden])
    f_biases_l1 = bias_variable([num_hidden])
    f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1), f_biases_l1))

    out_weights = weight_variable([num_hidden, num_labels])
    out_biases = bias_variable([num_labels])
    y_ = tf.nn.softmax(tf.matnul(f, out_weights)+out_biases)

    loss = -tf.reduce_sum(Y*tf.log(y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:
        tf.initialize_all_variables().run()
        for epoch in range(training_epochs):
            cost_history = np.empty(shape=[1], dtype=float)
            for b in range(total_batchs):
                offset = (b*batch_size) % (train_y.shape[0]-batch_size)
                batch_x = train_x[offset:(offset+batch_size), :, :, :]
                batch_y = train_y[offset:(offset+batch_size, :)]
                _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                cost_history = np.append(cost_history, c)
            print "Epoch: ", epoch, " Training Loss: ",np.mean(cost_history), " Training Accuracy: ", session.run(accuracy, feed_dict={X: train_x, Y: train_y})
        print "Testing Accuracy: ", session.run(accuracy, feed_dict={X: test_x, Y:test_y})



if __name__ == '__main__':
    input_height = 1
    input_width = 90
    num_labels = 2
    num_chanels = 4
    
    batch_size = 10
    kernel_size = 60
    depth = 60
    num_hidden = 1000

    learning_rate = 0.0001
    training_epochs = 5

    main()
