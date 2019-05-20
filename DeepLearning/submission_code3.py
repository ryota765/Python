
import math
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

lmd = 0.001 # L2正則化項の係数
n_epochs = 7
batch_size = 100
dropout_keep_prob = 0.5 # Dropout率

tf.reset_default_graph() # グラフのリセット

x_train, y_train, x_test = load_fashionmnist()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=1000)

x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool) # 訓練時orテスト時

class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x):
        self.W = tf.Variable(tf.random_uniform(shape=(in_dim, out_dim), minval=-0.08, maxval=0.08), name='W')
        self.b = tf.Variable(tf.zeros(out_dim), name='b')
        self.function = function
        
        self.params = [self.W, self.b]
    
    def __call__(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)
    
class Dropout:
    def __init__(self, dropout_keep_prob=1.0):
        self.dropout_keep_prob = dropout_keep_prob
        self.params = []
    
    def __call__(self, x):
        # 訓練時のみdropoutを適用
        return tf.cond(
            pred=is_training,
            true_fn=lambda: tf.nn.dropout(x, keep_prob=self.dropout_keep_prob),
            false_fn=lambda: x
        )

def sgd(cost, params, eta=0.1):
    grads = tf.gradients(cost, params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append(param.assign_sub(eta * grad))
    return updates

def adagrad(cost, params, eta=0.01, eps=1e-7):
    grads = tf.gradients(cost, params)
    updates = []
    for param, grad in zip(params, grads):
        G = tf.Variable(tf.zeros_like(param, dtype=tf.float32), name='G')
        updates.append(G.assign_add(grad**2))
        with tf.control_dependencies(updates):
            updates.append(param.assign_sub(eta / tf.sqrt(G + eps) * grad))
    return updates

def tf_log(x):
    return tf.log(tf.clip_by_value(x, 1e-10, x))

def compute_l2_reg(params):
    l2_reg = 0
    for param in params:
        l2_reg += tf.reduce_sum(tf.square(param)) # 2 * tf.nn.l2_lossを使っても良い
    return l2_reg

layers = [
    Dense(784, 200, tf.nn.relu),
    Dropout(dropout_keep_prob),
    Dense(200, 200, tf.nn.relu),
    Dropout(dropout_keep_prob),
    Dense(200, 10, tf.nn.softmax)
]

params = []
h = x
for layer in layers:
    h = layer(h)
    params += layer.params
y = h
l2_reg = compute_l2_reg(params)

cost = - tf.reduce_mean(tf.reduce_sum(t * tf_log(y), axis=1)) + lmd * l2_reg

updates = adagrad(cost, params)
train = tf.group(*updates)

n_batches = math.ceil(len(x_train) / batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
    x_train, y_train = shuffle(x_train, y_train)
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        sess.run(train, feed_dict={x: x_train[start:end], t: y_train[start:end], is_training: True})
    y_pred, cost_valid_ = sess.run([y, cost], feed_dict={x: x_valid, t: y_valid, is_training: False})
    print('EPOCH: {}, Valid Cost: {:.3f}, Valid Accuracy: {:.3f}'.format(
        epoch + 1,
        cost_valid_,
        accuracy_score(y_valid.argmax(axis=1), y_pred.argmax(axis=1))
    ))

y_pred = sess.run(y, feed_dict={x: x_test, is_training: False})
prediction = list(map(lambda x: np.argmax(x), y_pred))

submission = pd.Series(prediction, name='label')
submission.to_csv('/Users/ryotanomura/chap05/materials/submission_pred.csv', header=True, index_label='id')