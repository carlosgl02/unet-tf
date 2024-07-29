from utils.data_loading import *
from model_unet import unet_model as um
import tensorflow as tf
import logging
import wandb


dir_img = 'data/images'
dir_mask = 'data/masks'

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


# Función de entrenamiento del modelo
def train_model(model,
                dir_img,
                dir_mask,
                device,
                img_size = 200,
                epochs=20,
                batch_size=16,
                learning_rate=1e-4,
                patience=5,
                factor=0.5):



    # 1. Crear dataset
    dataset = load_dataset(dir_img, dir_mask, img_size)
    dataset_size = len(list(dataset))

    # 2. Dividir en particiones de entrenamiento y validación
    val_percent = 0.1
    n_val = int(dataset_size * val_percent)
    n_train = dataset_size - n_val

    # Dividir el dataset en train y validation
    train_dataset = dataset.skip(n_val)
    val_dataset = dataset.take(n_val)

    # Imprimir tamaños de los datasets
    print(f'Tamaño del dataset de entrenamiento: {n_train}')
    print(f'Tamaño del dataset de validación: {n_val}')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
    ''')

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice_coefficient])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience // 2,
                                                     min_lr=1e-6)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

    history = model.fit(
        train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE),
        validation_data=val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE),
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
    return history


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = '/gpu:0' if tf.config.list_physical_devices('GPU') else '/cpu:0'
    logging.info(f'Using device {device}')


    # Definir los argumentos (por simplicidad, se definen directamente en el código)
    class Args:
        epochs = 20
        batch_size = 16
        lr = 1e-4
        scale = 1.0
        val = 10  # porcentaje de validación
        amp = False
        load = None
        classes = 1
        bilinear = False


    args = Args()

    input_shape = (200, 200, 1)  # Tamaño de las imágenes de entrada
    model = um.build_unet(input_shape)

    logging.info(f'Network:\n'
                 f'\t{input_shape[-1]} input channels\n'
                 f'\t{args.classes} output channels (classes)\n')

    if args.load:
        model.load_weights(args.load)
        logging.info(f'Model loaded from {args.load}')

    try:
        with tf.device(device):
            train_model(
                model=model,
                dir_img=dir_img,
                dir_mask=dir_mask,
                device=device,
                img_size=200,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                patience=5,
                factor=0.5
            )
    except tf.errors.ResourceExhaustedError as e:
        logging.error('Detected ResourceExhaustedError! '
                      'Consider reducing the batch size or using mixed precision training (--amp)')
        tf.keras.backend.clear_session()
        # Intentar con menor batch size
        train_model(
            model=model,
            dir_img=dir_img,
            dir_mask=dir_mask,
            device=device,
            img_size=200,
            epochs=args.epochs,
            batch_size=args.batch_size // 2,
            learning_rate=args.lr,
            patience=5,
            factor=0.5
        )
