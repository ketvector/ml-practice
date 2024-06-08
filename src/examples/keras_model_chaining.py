import keras

def get_a_model():
    inputs = keras.layers.Input(shape=(4,))
    l1 = keras.layers.Dense(10)(inputs)
    l2 = keras.layers.ReLU()(l1)
    outputs = keras.layers.Dense(3, activation=keras.activations.sigmoid)(l2)

    return keras.Model(inputs=inputs, outputs=outputs)

def add_layer_to_model(model, layer):
    new_model_outputs = layer(model.layers[-1].output)
    new_model = keras.Model(inputs=model.inputs, outputs=new_model_outputs)
    return new_model 


def main():
    model1 = get_a_model()
    model1.summary()
    print(model1.get_weights())
    model2 = add_layer_to_model(model1, keras.layers.Dense(20))
    model2.summary()
    print(model2.get_weights())
    print("do model 2 layers point to same memory location as model1 ?", (model2.layers[0] is model1.layers[0]) and (model2.layers[1] is model1.layers[1]))

main()





