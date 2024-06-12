import keras_nlp
import keras


def get_model():
    name = "bert_tiny_en_uncased"
    backbone =  keras_nlp.models.BertBackbone.from_preset(
        name,
    )

    preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        name,
        sequence_length=512
    )
    backbone.trainable = False
    inputs = backbone.input
    sequence = backbone(inputs)["sequence_output"]
    for _ in range(2):
        sequence = keras_nlp.layers.TransformerEncoder(
            num_heads=2,
            intermediate_dim=512,
            dropout=0.1,
        )(sequence)
    # Use [CLS] token output to classify
    dense = keras.layers.Dense(5)(sequence[:, backbone.cls_token_index, :])
    outputs = keras.layers.Activation("sigmoid")(dense)

    model = keras.models.Model(inputs = inputs, outputs = outputs)
    return  preprocessor, model

def loss_fn(y_true, y_pred):
    return keras.ops.mean(-1 * (y_true * keras.ops.log(keras.ops.clip(y_pred, 1e-10, 1.0)) + (1.0 -y_true) * keras.ops.log(keras.ops.clip(1.0 - y_pred, 1e-10, 1.0))))