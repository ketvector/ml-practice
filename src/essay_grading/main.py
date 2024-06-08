import pandas as pd
from sklearn.model_selection import train_test_split
import keras_nlp
import keras

# Path to the CSV file
csv_file = './train.csv'

# Read the CSV file into a pandas dataframe
df = pd.read_csv(csv_file)
df["score"] = df["score"] - 1

# Get information about the dataframe
df_info = df.info()
# Print the information
print("dataframe info", df_info)


# Print the shape of the training set
print("Training set shape:", df.shape)


classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased",
    num_classes=6,
    activation=keras.activations.sigmoid,
)

model = classifier

print("model summary: ", model.summary())


train_df_subset = df

# Extract the "full_text" values as X
X = train_df_subset["full_text"].values

# Extract the "label" values as y
y = train_df_subset["score"].values

# Fit the classifier on the training data
model.fit(X, y, batch_size=50, epochs=2, validation_split=0.1)



