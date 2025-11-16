# --------------------------------------------------------------
# 🌼 Transfer Learning: Flower Classification with VGG16
# --------------------------------------------------------------

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# --------------------------------------------------------------
# 1️⃣ Load Dataset
# --------------------------------------------------------------
dataset, info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

class_names = info.features['label'].names
print("Classes:", len(class_names))

# --------------------------------------------------------------
# 2️⃣ Preprocess Data
# --------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    label = tf.one_hot(label, 102)
    return image, label

train_data = train_data.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_data = val_data.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --------------------------------------------------------------
# 3️⃣ Define the Model (VGG16 Transfer Learning)
# --------------------------------------------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False  # freeze pretrained layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(102, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# --------------------------------------------------------------
# 4️⃣ Train the Model
# --------------------------------------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# --------------------------------------------------------------
# 5️⃣ Evaluate and Visualize
# --------------------------------------------------------------
test_loss, test_acc = model.evaluate(test_data)
print(f"✅ Test Accuracy: {test_acc:.2f}")

# Plot Accuracy & Loss curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

# --------------------------------------------------------------
# 6️⃣ Predictions on sample images
# --------------------------------------------------------------
for images, labels in test_data.take(1):
    preds = model.predict(images)
    preds_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels.numpy(), axis=1)

plt.figure(figsize=(12,12))
for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(tf.cast(images[i], tf.uint8))
    plt.title(f"True: {class_names[true_classes[i]]}\nPred: {class_names[preds_classes[i]]}")
    plt.axis("off")
plt.show()