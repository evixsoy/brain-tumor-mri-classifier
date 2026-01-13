import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetB3

def create_unet_efficientnet(shape_input =(512,512,3), classes= 4, encoder_freeze = False, decoder_sizes = [512,256,128,64,32], dropout_rate=0.3):
    inputs = keras.Input(shape=shape_input, name='image_input')

    encoder = EfficientNetB3(
        include_top = False, #disable classification
        weights='imagenet', #pretrained
        input_tensor=inputs,
        pooling=None
    )

    if not encoder_freeze:
        #training whole model
        for layer in encoder.layers:
            layer.trainable = True
    else:
        # decoder + bottleneck - fine tuning
        for layer in encoder.layers:
            layer.trainable = False
    
    #get skip connections
    #256x256
    skip1 = encoder.get_layer('block1b_add').output
    #128x128
    skip2 = encoder.get_layer('block2c_add').output
    #64x64
    skip3 = encoder.get_layer('block3c_add').output
    #32x32
    skip4 = encoder.get_layer('block4e_add').output
    #16x16
    skip5 = encoder.get_layer('block6f_add').output
    
    #TODO Pridat exception kdyz to nenajde layery


    #bottleneck
    bridge = encoder.output
    bridge = layers.Conv2D(1024,kernel_size=3, padding='same',name ='bottleneck_conv1')(bridge)
    bridge = layers.BatchNormalization(name='bottleneck_bn1')(bridge)
    bridge = layers.Activation('relu', name='bottleneck_relu1')(bridge)

    bridge = layers.Conv2D(1024,kernel_size=3, padding='same',name ='bottleneck_conv2')(bridge)
    bridge = layers.BatchNormalization(name='bottleneck_bn2')(bridge)
    bridge = layers.Activation('relu', name='bottleneck_relu2')(bridge)

    bridge = layers.Dropout(dropout_rate, name='bottleneck_dropout')(bridge)


    #decoder
    skip_connections_list = [skip4, skip3, skip2, skip1]
    upsample = bridge

    #create decoder blocks
    for i in range(len(skip_connections_list)):
        upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name=f'decoder_upsample_{i+1}')(upsample)
        upsample = layers.Concatenate(name=f'decoder_concat_{i+1}')([upsample, skip_connections_list[i]])
        
        for g in range(2):
            upsample = layers.Conv2D(decoder_sizes[i], kernel_size=3, padding='same', name=f'decoder_conv{i+1}_{g+1}')(upsample)
            upsample = layers.BatchNormalization(name=f'decoder_bn{i+1}_{g+1}')(upsample)
            upsample = layers.Activation('relu', name=f'decoder_relu{i+1}_{g+1}')(upsample)
        
        upsample = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i+1}')(upsample)
    
    # Final upsampling to 512x512
    upsample = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='decoder_upsample_final')(upsample)
    upsample = layers.Conv2D(64, kernel_size=3, padding='same', name='decoder_conv_final_1')(upsample)
    upsample = layers.BatchNormalization(name='decoder_bn_final')(upsample)
    upsample = layers.Activation('relu', name='decoder_relu_final')(upsample)
    
    #output layer - multiclass
    outputs = layers.Conv2D(classes, kernel_size=1, activation = 'softmax', name='output', dtype='float32')(upsample)

    #create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='unet-efficientnetb3')
    
    return model
