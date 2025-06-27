import tensorflow as tf
import os

train_dir = './data/train'
validation_dir = './data/validation'


batch_size = 32
epochs = 100
img_height = 360
img_width = 640


train_ds = tf.keras.utils.image_dataset_from_directory(  # type: ignore
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    label_mode='int',
    shuffle=True,
    seed=123,
)

val_ds = tf.keras.utils.image_dataset_from_directory(  # type: ignore
    validation_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    label_mode='int',
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)  
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([  # type: ignore
    tf.keras.layers.RandomFlip("horizontal"),  # type: ignore
    tf.keras.layers.RandomRotation(0.1),  # type: ignore
    tf.keras.layers.RandomZoom(0.1),  # type: ignore
])

model = tf.keras.Sequential([  # type: ignore
    data_augmentation,
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # type: ignore

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # type: ignore
    tf.keras.layers.MaxPooling2D((2, 2)),  # type: ignore

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # type: ignore
    tf.keras.layers.MaxPooling2D((2, 2)),  # type: ignore

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),  # type: ignore
    tf.keras.layers.MaxPooling2D((2, 2)),  # type: ignore

    tf.keras.layers.Flatten(),  # type: ignore
    tf.keras.layers.Dense(128, activation='relu'),  # type: ignore
    tf.keras.layers.Dropout(0.5),  # type: ignore
    tf.keras.layers.Dense(3, activation='softmax')  # type: ignore
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

os.makedirs('model', exist_ok=True)
model.save('model/video_classifier.keras')




