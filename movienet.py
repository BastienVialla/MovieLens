from keras.models import load_model, model_from_json
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Activation, Reshape, Dropout
from keras.layers import Concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
from pathlib import Path
from sklearn.metrics import mean_squared_error

class MovieNet: 
    def rmse(self, y, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y)))

    def custom_activation(self, x):
        return K.sigmoid(x) * (self.max_rating+1)

    def __init__(self, n_users, n_movies, min_rating=0.5, max_rating=5):
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.n_users = n_users
        self.n_movies = n_movies
        
    def build_model(self, emb_size=[50, 50], hl=[10], drop=[0.25], emb_trainable=True):
        inputs = [Input(shape=(1,)), Input(shape=(1,))] #, Input(shape=(1,))]
        users_emb = Embedding(self.n_users, emb_size[0], name='users', trainable=emb_trainable)(inputs[0])
        movies_emb = Embedding(self.n_movies, emb_size[1], name='movies', trainable=emb_trainable)(inputs[1])
        outputs_emb = [Reshape(target_shape=(emb_size[0],))(users_emb), Reshape(target_shape=(emb_size[1],))(movies_emb)]
        
        output_model = Concatenate()(outputs_emb)
        for i in range(0, len(hl)):
            output_model = Dense(hl[i], kernel_initializer='uniform')(output_model)
            output_model = Activation('relu')(output_model)
            output_model = Dropout(drop[i])(output_model)

        output_model = Dense(1)(output_model)

        output_model = Activation(self.custom_activation)(output_model)
        
        self.model = KerasModel(inputs=inputs, outputs=output_model)
        
        opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        
        self.model.compile(loss='mse', optimizer=opt, metrics=[self.rmse])
        
          
    def prepare_input(self, _X):
        X = [_X.userId.values, _X.movieId.values]#, _X.ratingWeight]
        return X            
            
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
    
    def fit(self, X_train, y_train, X_valid, y_valid, epochs=50, batch_size=32, verbose=1):
        self.model.fit(self.prepare_input(X_train), y_train,
                       validation_data=(self.prepare_input(X_valid), y_valid),
                      epochs=epochs, batch_size=batch_size, verbose=verbose)
        # print("Result on validation data: ", self.evaluate(X_valid, y_valid))
        
    def predict(self, X):
        y_pred = self.model.predict(self.prepare_input(X))
        return y_pred.flatten()

    def save_model(self, path=Path(""), name="MovieModel"):
        self.model.save_weights(path/str(name+"_weights.h5"))
        with open(path/str(name+'_arch.json'), 'w') as f:
            f.write(self.model.to_json())
    
    def load_model(self, path=Path(""), name="MovieModel"):
        with open(path/str(name +'_arch.json'), 'r') as f:
            self.model = model_from_json(f.read(), custom_objects={"custom_activation": self.custom_activation})
        self.model.load_weights(path/str(name+"_weights.h5"))  