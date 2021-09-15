def cnn(n_factors):
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.activations import elu
    from tensorflow.keras.layers import BatchNormalization
    # Input block
    melgram_input= Input(shape = (48,48,1) , name = "Train_input")
    x = BatchNormalization(axis=1, name='bn_0_freq')(melgram_input)

    # Conv block 1
    x = Convolution2D(64, 2, 2, padding='same', name='conv1')(x)
    x = BatchNormalization(axis=3,  name='bn1')(x)
    x = elu(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 2, 2, padding='same', name='conv2')(x)
    x = BatchNormalization(axis=3, name='bn2')(x)
    x = elu(x)
    x = MaxPooling2D(pool_size=(2,2 ), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 2, 2, padding='same', name='conv3')(x)
    x = BatchNormalization(axis=3,  name='bn3')(x)
    x = elu(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)

    #output
    x = Flatten()(x)
    x = Dense(n_factors,activation='linear',name='output')(x)

    # Create model
    model = Model(melgram_input,x)
    return model
