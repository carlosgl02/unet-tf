import tensorflow as tf
from tensorflow.keras import layers

#Este bloque realiza dos convoluciones seguidas de activaciones ReLU, útil para extraer características de la imagen.
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.ReLU()(x)
    return x

#El bloque del codificador devuelve tanto las características extraídas (x) como la salida de la operación de pooling (p).
def encoder_block(input_tensor, num_filters):
    x = conv_block(input_tensor, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

#Este bloque aumenta la resolución de la imagen y combina características de diferentes niveles de abstracción
def decoder_block(input_tensor, concat_tensor, num_filters):
    x = layers.Conv2DTranspose(num_filters, 3, strides=2, padding='same')(input_tensor)

    # Ajustar dimensiones antes de concatenar
    h_diff = concat_tensor.shape[1] - x.shape[1]
    w_diff = concat_tensor.shape[2] - x.shape[2]

    if h_diff > 0 or w_diff > 0:
        x = layers.ZeroPadding2D(((h_diff // 2, h_diff - h_diff // 2),
                                  (w_diff // 2, w_diff - w_diff // 2)))(x)
    elif h_diff < 0 or w_diff < 0:
        x = layers.Cropping2D(((abs(h_diff) // 2, abs(h_diff) - abs(h_diff) // 2),
                               (abs(w_diff) // 2, abs(w_diff) - abs(w_diff) // 2)))(x)

    x = layers.concatenate([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x