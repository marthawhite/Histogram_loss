import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_model(model=None, pretrained=True, input_shape=(128, 128, 3)):
    
    if pretrained:
        weights = "imagenet"
    else:
        weights = None
    
    if model.lower() == "xception":
        base_model = keras.applications.Xception(
            include_top=False,
            weights=weights,
            input_tensor=layers.Input(input_shape),
            pooling="avg",
        )
        return base_model
    elif model.lower == "vgg16":
        base_model = keras.applications.VGG16(
            include_top=False,
            weights=weights,
            input_tensor=layers.Input(input_shape),
            pooling="avg",
        )
        return base_model
    elif model.lower == "vgg19":
        base_model = keras.applications.VGG19(
            include_top=False,
            weights=weights,
            input_tensor=layers.Input(input_shape),
            pooling="avg",
        )
        return base_model
    elif mode.lower == "resnet50":
        base_model = keras.applications.ResNet50V2(
            include_top=False,
            weights=weights,
            input_tensor=layers.Input(input_shape),
            pooling="avg",
        )
        return base_model
    elif mode.lower == "resnet101":
        base_model = keras.applications.ResNet101V2(
            include_top=False,
            weights=weights,
            input_tensor=layers.Input(input_shape),
            pooling="avg",
        )
        return base_model
    elif mode.lower == "resnet152":
        base_model = keras.applications.ResNet152V2(
            include_top=False,
            weights=weights,
            input_tensor=layers.Input(input_shape),
            pooling="avg",
        )
        return base_model
    else:
        assert False
        