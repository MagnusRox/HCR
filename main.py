import datetime
import math

import pandas as pd
import tensorflow as tf
import model_settings as ms
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder


def preprocess_image(path, text, image_size):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False, dtype=tf.uint8)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, image_size)
        return image, text

    except tf.errors.InvalidArgumentError:
        # Skip the file if it has an unknown image file format
        print(f"Skipping file: {path}. Unknown image file format.")
        return None, None


def create_cnn_model(input_shape, num_classes, optimizer , loss_fn):
    layers = tf.keras.layers

    # Input layer
    input_data = layers.Input(shape=input_shape, name='input_data')

    # Convolutional layers
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_data)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu')(pool2)
    conv4 = layers.Conv2D(128, (3, 3), activation='relu')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(256, (3, 3), activation='relu')(pool3)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu')(conv5)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv6)

    # Flatten layer
    flatten = layers.Flatten()(pool4)

    # Fully connected layers
    fc1 = layers.Dense(512, activation='relu')(flatten)
    dropout = layers.Dropout(0.5)(fc1)
    fc2 = layers.Dense(256, activation='relu')(dropout)

    # Output layer
    output = layers.Dense(num_classes, activation='softmax')(fc2)

    model = tf.keras.Model(inputs=input_data, outputs=output, name='CNN_Model')

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Create the model

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    return model, early_stopping, tensorboard_callback


#remove_corrupt_files()

data = pd.read_csv(r'Post-processing/processed-corrupt-removed.csv', index_col=False)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data['text'] = data['text'].str.lower()
data = data.dropna()
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['text'])
data['label'] = data['label'].astype(int)  # Convert label column to integer type
vocab_size = len(data["label"].unique())
dataset = tf.data.Dataset.from_tensor_slices((data['path'].astype('string'), data['label']))

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=len(data))
# Define the split percentages
train_split = 0.7
val_split = 0.15
test_split = 0.15

# Calculate the sizes of each split
num_samples = len(data)
num_train_samples = int(train_split * num_samples)
num_val_samples = int(val_split * num_samples)
num_test_samples = num_samples - num_train_samples - num_val_samples

# Split the dataset
train_dataset = dataset.take(num_train_samples)
val_dataset = dataset.skip(num_train_samples).take(num_val_samples)
test_dataset = dataset.skip(num_train_samples + num_val_samples).take(num_test_samples)

# Preprocess and batch the datasets
batch_size = ms.model_config['batch']
image_size = ms.model_image_size
preprocessed_train_dataset = train_dataset.map(lambda path, label: preprocess_image(path, label, image_size)).batch(
    batch_size)
preprocessed_val_dataset = val_dataset.map(lambda path, label: preprocess_image(path, label, image_size)).batch(
    batch_size)
preprocessed_test_dataset = test_dataset.map(lambda path, label: preprocess_image(path, label, image_size)).batch(
    batch_size)
# Create the CNN model

epochs = ms.model_config['epochs']
# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=ms.model_config['learning_rate'])
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)


model, early_stopping, tensorboard_callback = create_cnn_model(input_shape=(360, 160, 3), num_classes=vocab_size, optimizer=optimizer, loss_fn=loss_fn)

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()


max_iterations = len(preprocessed_train_dataset)+1
#max_iterations = 5

for epoch in range(epochs):
    iterator = iter(preprocessed_train_dataset)
    print("\nStart of epoch %d" % (epoch,))
    # Iterate over the batches of the dataset.
    i = 0
    while i < max_iterations:
        print("Currently running {} batch".format(i))
        try:
            i = i + 1
            x_batch_train, y_batch_train = next(iterator)
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if i % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (i, float(loss_value))
                )
                print("Seen so far: %s samples" % ((i + 1) * batch_size))

            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            for x_batch_val, y_batch_val in preprocessed_val_dataset:
                val_logits = model(x_batch_val, training=False)
                # Update val metrics
                val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            print("Validation acc: %.4f" % (float(val_acc),))
        except Exception as e:
            print(e)
            continue

# Evaluate the model
test_loss, test_accuracy = model.evaluate(preprocessed_test_dataset)
