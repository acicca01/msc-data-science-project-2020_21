def cnn(bands,frames,n_factors):
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.activations import elu,relu
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.initializers import he_normal
    # Input block
    melgram_input= Input(shape = (bands,frames,1) , name = "Train_input")
    #x = BatchNormalization(axis=3, name='bn_0_freq')(melgram_input)

    # Conv block 1
    x = Convolution2D(32, 3, padding='same', name='conv1',kernel_initializer=he_normal())(melgram_input)
    x = BatchNormalization(axis=3,  name='bn1')(x)
    x = relu(x)
    x = MaxPooling2D(pool_size=(1, 4), name='pool1')(x)
    x = Dropout(0.5)(x)
    # Conv block 2
    x = Convolution2D(64, 3, padding='same', name='conv2')(x)
    x = BatchNormalization(axis=3, name='bn2')(x)
    x = relu(x)
    x = MaxPooling2D(pool_size=(2,4 ), name='pool2')(x)
    x = Dropout(0.5)(x)

    # Conv block 3
    x = Convolution2D(128, 3, padding='same', name='conv3')(x)
    x = BatchNormalization(axis=3,  name='bn3')(x)
    x = relu(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)
    x = Dropout(0.5)(x)

    # Conv block 4
    x = Convolution2D(256, 3, padding='same', name='conv4')(x)
    x = BatchNormalization(axis=3,  name='bn4')(x)
    x = relu(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)
    x = Dropout(0.5)(x)

    # Conv block 5
    x = Convolution2D(512, 3, padding='same', name='conv5')(x)
    x = BatchNormalization(axis=3,  name='bn5')(x)
    x = relu(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)
    x = Dropout(0.5)(x)

    #output
    x = Flatten()(x)
    x = Dense(256,activation='relu',name='dense')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_factors,activation='linear',name='output')(x)

    # Create model
    model = Model(melgram_input,x)
    return model
