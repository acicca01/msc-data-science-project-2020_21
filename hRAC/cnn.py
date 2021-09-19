def cnn(bands,frames,n_factors):
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
    import tensorflow as tf
    import tensorflow.keras.backend as K
    from tensorflow.keras.activations import elu
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.regularizers import l2 
    from tensorflow.keras.initializers import he_normal
    # Input block
    melgram_input= Input(shape = (bands,frames,1) , name = "Train_input")
    x = BatchNormalization(axis=3, name='bn_0_freq')(melgram_input)

    # Conv block 1
    x = Convolution2D(32, 2, 2, padding='same', name='conv1',kernel_initializer=he_normal)(x)
    x = BatchNormalization(axis=3,  name='bn1')(x)
    x = elu(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool1')(x)
    x = Dropout(0.2)(x)
    # Conv block 2
    x = Convolution2D(64, 2, 2, padding='same', name='conv2',kernel_regularizer=l2(l=0.01))(x)
    x = BatchNormalization(axis=3, name='bn2')(x)
    x = elu(x)
    x = MaxPooling2D(pool_size=(1,2 ), name='pool2')(x)
    x = Dropout(0.2)(x)

    # Conv block 3
    x = Convolution2D(64, 1, 2, padding='same', name='conv3',kernel_regularizer=l2(l=0.01))(x)
    x = BatchNormalization(axis=3,  name='bn3')(x)
    x = elu(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool3')(x)
    x = Dropout(0.2)(x)

    # Conv block 4
    x = Convolution2D(64, 1, 2, padding='same', name='conv4', kernel_regularizer=l2(l=0.01))(x)
    x = BatchNormalization(axis=3,  name='bn4')(x)
    x = elu(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool4')(x)
    x = Dropout(0.2)(x)

    # Conv block 5
    x = Convolution2D(64, 2, 2, padding='same', name='conv5', kernel_regularizer=l2(l=0.01))(x)
    x = BatchNormalization(axis=3,  name='bn5')(x)
    x = elu(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool5')(x)
    x = Dropout(0.2)(x)

    #output
    x = Flatten()(x)
    x = Dense(48, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(n_factors,activation='linear',name='output')(x)

    # Create model
    model = Model(melgram_input,x)
    return model
