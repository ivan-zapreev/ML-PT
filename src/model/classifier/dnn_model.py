import functools

from src.utils.logger import logger

from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler

# The basic model
def __create_basic_model(num_inputs, emb_dim):
    # create model
    model = Sequential()
    model.add(Dense(emb_dim, input_shape=(num_inputs,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def instantiate_dnn_model(num_inputs, emb_dim, num_epochs, batch_size, verbose):
    # Create pipeline
    create_basic_model = functools.partial(__create_basic_model, num_inputs=num_inputs, emb_dim=emb_dim)
    estimators = [('standardize', StandardScaler()), \
                  ('mlp', KerasClassifier(model=create_basic_model, epochs=num_epochs, batch_size=batch_size, verbose=verbose))]
    pipeline = Pipeline(estimators)

    return pipeline