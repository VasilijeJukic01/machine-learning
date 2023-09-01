import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import CSVLogger

# Logger
log_path = 'training_log.csv'
csv_logger = CSVLogger(log_path)

# Inputs
input_path = 'SaYoPillow.csv'

input_data = pd.read_csv(input_path, sep=',', dtype='float')
print(input_data)

trainData, testData, trainLabel, testLabel = \
    train_test_split(input_data.iloc[:, :8], input_data['stress_level'], test_size=0.2)

# One hot encoding
num_classes = 5
trainLabel = to_categorical(trainLabel, num_classes)
testLabel = to_categorical(testLabel, num_classes)

# Heatmap
plt.figure(figsize=(15, 8))
sb.heatmap(trainData.corr(), annot=True, cmap="Greens")
plt.show()

# MLP Model
model = Sequential()
model.add(Dense(50, activation="sigmoid"))
model.add(Dense(50, activation="sigmoid"))
model.add(Dense(5, "softmax"))

epochs = 50
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
stats = model.fit(trainData, trainLabel, epochs=epochs, validation_split=0.2, callbacks=[csv_logger])

# Plot
stats_df = pd.DataFrame(stats.history)
stats_df['epoch'] = list(range(epochs))
plt.figure(figsize=(10, 8))
sb.lineplot(y='loss', x='epoch', data=stats_df, color='red', linewidth=2.5, label="Training loss")
sb.lineplot(y='val_loss', x='epoch', data=stats_df, color='blue', linewidth=2.5, label="Validation loss")
plt.grid()
plt.legend()
plt.title("Training and validation loss")
plt.xticks(range(0, 51, 5))
plt.show()

with open('mlp_model.pkl', 'wb') as f:
    pickle.dump(model, f)

