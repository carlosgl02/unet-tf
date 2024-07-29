import tensorflow as tf
import os


def load_dataset(dir_img, dir_mask, img_size):
    def load_images_and_masks():
        images = sorted([os.path.join(dir_img, f) for f in os.listdir(dir_img) if f.endswith('.jpg')])
        masks = sorted([os.path.join(dir_mask, f) for f in os.listdir(dir_mask) if f.endswith('.jpg')])
        return images, masks

    def preprocess(image, mask):
        image = tf.image.resize(image, [img_size, img_size])
        mask = tf.image.resize(mask, [img_size, img_size * 2])  # Ajuste para 200x400
        image = image / 255.0
        mask = mask / 255.0
        return image, mask

    def load_image(image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        return image, mask

    images, masks = load_images_and_masks()
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(lambda image, mask: tf.py_function(load_image, [image, mask], [tf.float32, tf.float32]))
    dataset = dataset.map(preprocess)

    return dataset