import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) < version.parse("2.8"):
    from tensorflow.keras.layers.experimental.preprocessing import Normalization
elif version.parse(tf.__version__) < version.parse("2.13"):
    from tensorflow.keras.layers import Normalization
else:
    Normalization = tf.keras.layers.Normalization


def get_preprocessing_layer():
    """Return preprocessing layer for EfficientNet Lite variants."""
    return Normalization(
        mean=(127.0, 127.0, 127.0),
        variance=(128.0**2, 128.0**2, 128.0**2),
        axis=3 if tf.keras.backend.image_data_format() == "channels_last" else 1,
    )
