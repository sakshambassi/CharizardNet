from keras.layers import Dense, Input


class Encoder:
    def __init__(self):
        pass

    def encode(self):
        pass

    def input_dims(self):
        pass


class FcEncoder(Encoder):
    def __init__(self):
        pass

    def encode(self, inputs_dims):
        inputs = Input(shape=inputs_dims, name='input')
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
        return inputs, x

    def reshape_input(self, x):
        return x.reshape(x.shape[0], -1)

    def input_dims(self, x):
        return (x.shape[1],)
