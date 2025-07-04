import tensorflow as tf
import os
import signal
import sys

# === 경로 설정 ===
train_dir = './data/train'
validation_dir = './data/validation'
checkpoint_dir = './model/checkpoints'
checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint_latest.keras')
epoch_info_file = os.path.join(checkpoint_dir, 'last_epoch.txt')
os.makedirs(checkpoint_dir, exist_ok=True)

# === 하이퍼파라미터 ===
batch_size = 32
epochs = 100
img_height = 360
img_width = 640

# === 데이터 불러오기 ===
train_ds = tf.keras.utils.image_dataset_from_directory( #type: ignore
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    label_mode='int',
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory( #type: ignore
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

# === 데이터 증강 ===
data_augmentation = tf.keras.Sequential([ #type: ignore
    tf.keras.layers.RandomFlip("horizontal"), #type: ignore
    tf.keras.layers.RandomRotation(0.1), #type: ignore
    tf.keras.layers.RandomZoom(0.1), #type: ignore
    tf.keras.layers.RandomContrast(0.1), #type: ignore
    tf.keras.layers.RandomBrightness(0.1), #type: ignore
    tf.keras.layers.RandomTranslation(0.1, 0.1), #type: ignore
])

# === 모델 구성: MobileNetV2 기반 ===
base_model = tf.keras.applications.MobileNetV2( #type: ignore
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # 학습 초반에는 freeze

model = tf.keras.Sequential([  #type: ignore
    data_augmentation,
    tf.keras.layers.Rescaling(1./255), #type: ignore
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(), #type: ignore
    tf.keras.layers.Dense(128, activation='relu'), #type: ignore
    tf.keras.layers.Dropout(0.5), #type: ignore
    tf.keras.layers.Dense(3, activation='softmax')  # LOL, TFT, Unknown  #type: ignore
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === 체크포인트 불러오기 ===
initial_epoch = 0
if os.path.exists(checkpoint_path):
    print("🔄 기존 체크포인트를 찾았습니다. 학습을 이어서 진행합니다...")
    model = tf.keras.models.load_model(checkpoint_path) #type: ignore
    if os.path.exists(epoch_info_file):
        try:
            with open(epoch_info_file, 'r') as f:
                initial_epoch = int(f.read().strip())
                print(f"🔄 에포크 {initial_epoch + 1}부터 학습을 재개합니다.")
        except:
            print("⚠️ 에포크 정보를 읽을 수 없습니다. 처음부터 시작합니다.")
else:
    print("🆕 새로운 모델로 학습을 시작합니다.")

# === Ctrl + C : 안전하게 저장 ===
current_epoch = initial_epoch

def signal_handler(sig, frame):
    global current_epoch
    print('\n🛑 학습 중단 감지! 현재 상태 저장 중...')
    model.save(checkpoint_path)
    with open(epoch_info_file, 'w') as f:
        f.write(str(current_epoch))
    print(f"✅ 체크포인트 저장 완료: {checkpoint_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class SaveOnInterrupt(tf.keras.callbacks.Callback): #type: ignore
    def __init__(self, checkpoint_path, epoch_info_file):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.epoch_info_file = epoch_info_file

    def on_epoch_end(self, epoch, logs=None):
        global current_epoch
        current_epoch = epoch + 1
        with open(self.epoch_info_file, 'w') as f:
            f.write(str(current_epoch))

    def on_train_end(self, logs=None):
        self.model.save(self.checkpoint_path)
        print(f"✅ 학습 완료! 모델 저장됨: {self.checkpoint_path}")

# === 콜백 구성 ===
callbacks = [
    SaveOnInterrupt(checkpoint_path, epoch_info_file),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1), #type: ignore
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) #type: ignore
]

# === 학습 시작 ===
print(f"🚀 학습 시작 (에포크 {initial_epoch + 1}부터 {epochs}까지)")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    verbose=2
)

# === 최종 모델 저장 ===
os.makedirs('model', exist_ok=True)
model.save('model/video_classifier.keras')
print("✅ 최종 모델! : model/video_classifier.keras")
