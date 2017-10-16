import pandas as pd
import numpy as np

path = "./saved_models/model"

### For the (Clean and convert data)

# np.set_printoptions(threshold=np.inf) # display full numpy array
pd.set_option('display.expand_frame_repr', False)  # Widen the print display to display more columns

# Read train_data
df_train = pd.read_csv('all_data_by_day.csv')
# print(df_train) to check the df

# Remove rows with NaN values and Convert in to float
df_train.dropna(how='any', inplace=True)
df_train = df_train.astype(float, errors='ignore')

# Get x and y data (if we don't use pd.DataFrame, it will be in series)
location_train = pd.DataFrame(df_train.iloc[:, 0:2])
x_train_df = df_train.iloc[:, 4:len(df_train.columns)]
y_train_df = pd.DataFrame(df_train.iloc[:, 2] - df_train.iloc[:, 3], columns=['actual_activity'])
var_size = x_train_df.shape[1]

# Convert into np.array to feed into tensor
x_data = x_train_df.as_matrix(columns=None)
y_data = y_train_df.as_matrix(columns=None).reshape(len(y_train_df), 1)

# # For testing only
# # Read test_data
# df_test = pd.read_csv('test_data.csv')
# # print(df_test) to check the df
#
# # Remove rows with NaN values and Convert in to float
# df_test.dropna(how='any', inplace=True)
# df_test = df_test.astype(float, errors='ignore')
#
# # Get x and y data (if we don't use pd.DataFrame, it will be in series)
# location_test = pd.DataFrame(df_test.iloc[:, 0:2])
# x_test_df = df_test.iloc[:, 4:len(df_test.columns)]
# y_test_df = pd.DataFrame(df_test.iloc[:, 2] - df_test.iloc[:, 3], columns=['actual_activity'])
#
# # Convert into np.array to feed into tensor
# x_test_data = x_test_df.as_matrix(columns=None)
# y_test_data = y_test_df.as_matrix(columns=None).reshape(len(y_test_df), 1)


### For the model

import tensorflow as tf
import matplotlib.pyplot as plt
# import numpy as np
import csv

tf.set_random_seed(1)

# Def parameters
global_step = tf.Variable(0, trainable=False)  # Initial value = 0, cant be trained
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                           global_step,
                                           500,  # decay_step
                                           0.20,  # decay_rate
                                           staircase=True)  # Staircase decay
phase = tf.placeholder(tf.bool, name='phase')  # Give the phase value 0 or 1
keep_prob = tf.placeholder(tf.float32)
training_epochs = 4000

# Def tf Graph input
x_input = tf.placeholder("float", [None, var_size])
y_input = tf.placeholder("float", [None, 1])


# Define fully connected layer
def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size,
                                             activation_fn=None,
                                             scope=scope)


def dense_batch(x, phase, size, scope, keep_prob):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, size,
                                               activation_fn=None,
                                               scope='dense')
        drop = tf.nn.dropout(h1, keep_prob)
        return tf.nn.relu(drop, 'sigmoid')


# Construct model Neural Network with 3 hidden layers
hl1 = dense_batch(x_input, phase, 1000, 'layer1', keep_prob)
hl2 = dense_batch(hl1, phase, 600, 'layer2', keep_prob)
hl3 = dense_batch(hl2, phase, 300, 'layer3', keep_prob)
y_pred = dense(hl3, 1, 'pred')

# Define pred, loss, error
with tf.name_scope('y_p'):
    y_p = y_pred
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.square(y_pred - y_input))
with tf.name_scope('ave_error'):
    error = (tf.abs(y_pred - y_input) / y_input)
    ave_error = tf.reduce_mean(error)

# Create a model saver
model_saver = tf.train.Saver()

# Add optimizer (Adam)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Ensure to update before train_step
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

# Def initialization of global var
init = tf.global_variables_initializer()

# Def training cycle
def train_model(x_train, y_train, x_test, y_test, save_model, show_graph, seed=0):
    sess = tf.Session()
    sess.run(init)
    history = []

    # Input feed_dict
    train_d = {x_input: x_train, y_input: y_train, 'phase:0': 1, keep_prob: 0.92} # Just to train the optimizer
    train_acc_d = {x_input: x_train, y_input: y_train, 'phase:0': 0, keep_prob: 1.0} # Use the model on the train set
    test_d = {x_input: x_test, y_input: y_test, 'phase:0': 0, keep_prob: 1.0} # Use the model on the test set

    for i in range(training_epochs):
        # Run optimizer
        sess.run(optimizer, feed_dict=train_d)
        train = sess.run(ave_error, feed_dict=train_acc_d)
        test = sess.run(ave_error, feed_dict=test_d)
        y_predicted = sess.run(y_p, feed_dict=test_d)

        # print("train acc: {:.6f}, test acc: {:.6f}, i: {}".format(1 - train, 1 - test, i))
        history.append([i + 1, 1 - train, 1 - test, y_test, y_predicted])

    if save_model:
        model_saver.save(sess, path)  # Save model

    if show_graph:
        g = list(map(lambda x: x[0], history))
        [train_acc] = plt.plot(g, list(map(lambda x: x[1], history)), label='train accuracy', alpha=0.7) # Alpha to make it transparent
        [test_acc] = plt.plot(g, list(map(lambda x: x[2], history)), label='test accuracy') # We need the 'comma' because x and y has mant values and it unpack and use each value
        plt.legend(handles=[train_acc, test_acc])
        plt.axis([0, training_epochs, 0, 1])
        plt.show()

    return history[len(history) - 1] # Return that last one


# This is to predict new data
def use_model(x_test_data):
    sess = tf.Session()
    sess.run(init)
    model_saver.restore(sess, path)  # restore model
    pred = sess.run(y_pred, feed_dict={x_input: x_test_data, 'phase:0': 0, keep_prob: 1.0})
    return list(pred)

if __name__ == '__main__':
    with open('LOOCV_results.csv', 'w+') as fout:
        w = csv.writer(fout)
        # w.writerow(["accuracy", "test item", "train item w/max error"])
        result_all = pd.DataFrame()
        result_test_all = pd.DataFrame()
        for i, (x, y) in enumerate(zip(x_data, y_data)): # Make iteration or counter for each row of x,y starting from 0
            x_train = np.append(x_data[:i], x_data[i + 1:], 0) # axis = 0 means putting it on the bottom, axis = None means flattening it
            y_train = np.append(y_data[:i], y_data[i + 1:], 0) # Leave one row out cross validation, :i means from begin to i-1, i+1: means from i+1 to end -> leave out i
            x_test = (x_data[i]).reshape(1, var_size)
            y_test = (y_data[i]).reshape(1, 1)
            result_i = train_model(x_train, y_train, x_test, y_test, save_model=True, show_graph=False)
            # print('LOOCV: {:<3} Train_acc: {:<8.3f} Test_acc: {:<8.3f} y_actual: {} y_predicted: {}'.format(
            #     i, result_i[1], result_i[2], result_i[3], result_i[4])) # This print is to show the progress

            result_i_list = (result_i[1], result_i[2], result_i[3], result_i[4]) # put all into a list (int, int, ndarray, ndarray)
            result_i_df = pd.DataFrame(np.array(result_i_list).reshape(-1, len(result_i_list)), columns=[
                        'Train_acc', 'Test_acc', 'y_actual', 'y_predicted']) # convert all into ndarray then convert into df
            result_all = result_all.append(result_i_df, ignore_index=True) # append into final result__all_df
            print(result_all)
            w.writerow(result_i_list)

            # # For test_data
            # result_test_i = use_model(x_test_data)
            # result_test_df_i = pd.DataFrame(np.array(result_test_i).reshape(len(result_test_i), -1), columns=[
            #                 'prediction'])
            # result_test_all_i = pd.concat([location_test, y_test_df, result_test_df_i], axis=1)
            # result_test_all = result_test_all.append(result_test_all_i, ignore_index=False)  # append into final result__all_df
            # print(result_test_all)

            fout.flush()
    print(result_all)
    result_all.to_csv('result_all.csv', index=False)
    result_test_all.to_csv('result_test_all.csv', index=False)
