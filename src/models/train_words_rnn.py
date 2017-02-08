# LSTM and CNN for sequence classification in the IMDB dataset
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import cPickle as pickle
from keras.callbacks import EarlyStopping
from models.down_sampling import balanced_subsample

# fix random seed for reproducibility
numpy.random.seed(7)
# keep the top n words, zero the rest
top_item = 5005
# truncate and pad input sequences
max_review_length = 60


X,y = pickle.load(open("../feature_engg/feature/word_features.p", "r") ), pickle.load(open("../feature_engg/feature/labels.p", "r") )
# X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8)
X_train, X_test, y_train, y_test = X,X[0:1000],y,y[0:1000]


X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_item, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100, dropout_W=0.1, dropout_U=0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

num_epoch=20
batch_size=64
verbose = True

call_ES = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max')

print "len before down sampling", len(X_train)
X_train, y_train = balanced_subsample(X_train, y_train, subsample_size = 2.5, possible_y=['1','0'] )
print "len after down sampling", len(X_train)

model.fit(X_train, y_train, nb_epoch=num_epoch, batch_size=batch_size,
          verbose=verbose, callbacks=[call_ES], validation_split=0.1)

model.save("./model/rnn_words.h5", overwrite=True)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))