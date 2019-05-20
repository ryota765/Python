from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from mlp_connector import Connector
import numpy as np

con = Connector()
cnt_list,label_sub = con.return_ids('onayami.csv')

label = []
category = ['質問板', '育児', '心の悩み', 'おしゃべり板', '家庭・家族', '友人', 'その他', '身体', '職場・仕事', '学校', '恋愛/17才', '恋愛/29才', '恋愛/30才', '50才全般', '性の悩み']
for elem in label_sub:
    try:
        label.append(category.index(elem))
    except ValueError:
        print(elem)

label = np.array(label)
cnt_list = np.array(cnt_list)

nb_classes = 15  #カテゴリ数
batch_size = 128  #バッチサイズ
nb_epoch = 30  #学習回数

def build_model():
    global max_words
    model = Sequential()
    #model.add(Dense(32, input_shape=(max_words,),kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(512, input_shape=(max_words,)))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

max_words = len(cnt_list[0])

x_train, x_test, y_train, y_test = train_test_split(cnt_list, label)
print('xtrain',x_train[0:3])

y_train = np_utils.to_categorical(y_train, nb_classes)

print('ytrain', y_train[0:100])
#model = KerasClassifier(build_fn=build_model,nb_epoch=nb_epoch,batch_size=batch_size)
model = KerasClassifier(build_fn=build_model)
model.fit(x_train,y_train,verbose=1,nb_epoch=nb_epoch,batch_size=batch_size)

y = model.predict(x_test)
ac_score = metrics.accuracy_score(y_test,y)
print('acc={}'.format(ac_score))
print('dense=512, sigmoid')
