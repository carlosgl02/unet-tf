from .unet_parts import *
from tensorflow.keras import layers, models

def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    #Se aplican bloques del codificador en secuencia, con un aumento en el número de filtros en cada paso.
    #encX: Características extraídas en cada nivel.
    #pX: Salidas de pooling para el siguiente nivel.

    enc1, p1 = encoder_block(inputs, 64)
    enc2, p2 = encoder_block(p1, 128)
    enc3, p3 = encoder_block(p2, 256)
    enc4, p4 = encoder_block(p3, 512)

    # Bridge
    #Aplica el bloque de convolución a la salida del último bloque del codificador, actuando como un puente entre el codificador y el decodificador
    b = conv_block(p4, 1024)

    # Decoder
    #Se aplican bloques del decodificador en secuencia, con una disminución en el número de filtros en cada paso.
    #Cada bloque del decodificador recibe como entrada las características del nivel correspondiente del codificador
    #(skip connections) para combinar información de diferentes resoluciones.

    dec4 = decoder_block(b, enc4, 512)
    dec3 = decoder_block(dec4, enc3, 256)
    dec2 = decoder_block(dec3, enc2, 128)
    dec1 = decoder_block(dec2, enc1, 64)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(dec1)

    # Redimiensionamiento para obtener 200x400
    outputs = layers.Resizing(200, 400)(outputs)

    model = models.Model(inputs, outputs)

    return model