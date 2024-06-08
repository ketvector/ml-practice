import keras

tvl = keras.layers.TextVectorization(max_tokens=4000, output_mode='multi_hot')
tvl.adapt(["that is a great purse", "purse is beautiful", "beautiful and great"])

print("vocabulary", tvl.get_vocabulary())

response = tvl(["great purse is lost", "beautiful flower on the purse"])
print("sample layer output", response)