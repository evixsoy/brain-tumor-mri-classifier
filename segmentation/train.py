import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import keras.backend as K
import albumentations as alb

from unet import create_unet_efficientnet

# changeable values
DATASET_FOLDER = 'dataset_split_segmentation'

MODEL = 'efficientnetb3'
INPUT_SIZE = (512,512)
CLASSES = 4
CLASS_NAMES = ['background', 'glioma', 'meningioma', 'pituitary']

BATCH_SIZE = 1
EPOCHS = 100 # with EarlyStopping

# Optimized memory management
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False) # Disable XLA JIT

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU Setup Error: {e}")

#setup augmentation
training_augmentation = alb.Compose([
    alb.HorizontalFlip(p=0.5),
    alb.Rotate(limit=25, p=0.7),
    alb.Affine(scale=0.15, translate_percent=0.1, rotate=25, p=0.5),

    alb.ElasticTransform(alpha=0.5, sigma=30, p=0.2),
    alb.GridDistortion(num_steps=5, distort_limit=0.3, p=0.15),

    #image only
    alb.RandomBrightnessContrast(p=0.5),
    alb.RandomGamma(gamma_limit=(80, 120), p=0.5),

    alb.GaussianBlur(blur_limit=(3,7), p=0.5),

    alb.CLAHE(clip_limit=4.0, p=0.5),

    alb.CoarseDropout(num_holes_range=(1,4), hole_height_range=(8,16), hole_width_range=(8,16), p=0.1)
])

validation_augmentation = alb.Compose([])


class LazySegmentationGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_folder_path, mask_folder_path, batch_size, classes, 
                 input_size=(512,512), augmentation=None, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.image_folder_path = image_folder_path
        self.mask_folder_path = mask_folder_path
        self.batch_size = batch_size
        self.classes = classes
        self.input_size = input_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        
        # Get list of all image paths
        self.image_paths = []
        self.mask_paths = []
        self.tumor_labels = []
        
        tumor_types = ['glioma', 'meningioma', 'pituitary']
        for tumor_index, tumor_type in enumerate(tumor_types):
            img_folder = os.path.join(image_folder_path, tumor_type)
            mask_folder = os.path.join(mask_folder_path, tumor_type)
            
            for file in sorted(os.listdir(img_folder)):
                self.image_paths.append(os.path.join(img_folder, file))
                self.mask_paths.append(os.path.join(mask_folder, file))
                self.tumor_labels.append(tumor_index + 1)  # 1=glioma, 2=meningioma, 3=pituitary
        
        self.indices = np.arange(len(self.image_paths))
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_masks = []
        
        for i in batch_indices:
            image = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = image / 255.0
            
            mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
            mask = mask * self.tumor_labels[i]
            
            # Apply augmentation
            if self.augmentation:
                augmented = self.augmentation(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            batch_images.append(image)
            batch_masks.append(mask)
        
        batch_images = np.array(batch_images, dtype=np.float32)
        batch_masks = np.array(batch_masks, dtype=np.float32)
        
        # Convert masks to one-hot encoding
        batch_masks_onehot = to_categorical(batch_masks, num_classes=self.classes)
        
        return batch_images, batch_masks_onehot
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


model = create_unet_efficientnet(
    shape_input=(512, 512, 3), 
    classes=CLASSES,
    encoder_freeze=False,
    decoder_sizes=[128, 64, 32, 16],  # Reduced to save VRAM at 512x512
    dropout_rate=0.3
)
print(f"Model parameters: {model.count_params():,}")


#loss funkce
def dice_sum(y_true, y_pred, smooth = 1e-6):
    # convert na float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    dice_sum = 0.0
    
    for class_idx in range(4):
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        
        y_true_f = tf.reshape(y_true_class, [-1])
        y_pred_f = tf.reshape(y_pred_class, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_sum += dice
    return dice_sum

def combined_loss(y_true, y_pred,):
    #dice loss
    dice_total = dice_sum(y_true, y_pred)
    dice_mean = dice_total / 4.0
    dice_loss =  1.0 - dice_mean

    #focal loss
    gamma=2.0
    alpha=0.25
    epsilon = 1e-7

    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    ce = -y_true * tf.math.log(y_pred)
    focal_weight = alpha * tf.pow(1 - y_pred, gamma)
    focal_loss = focal_weight * ce

    focal_loss = tf.reduce_sum(focal_loss, axis=-1)
    focal_loss = tf.reduce_mean(focal_loss)
    
    return 0.5 * dice_loss + 0.5 * focal_loss


#metrics
def dice_coefficient_multiclass(y_true, y_pred):
    return dice_sum(y_true, y_pred) / 4.0

def dice_coefficient_per_class(class_idx, class_name):
    def dice(y_true, y_pred, smooth=1e-6):
        # Cast to float32 to match types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        y_true_class = y_true[..., class_idx]
        y_pred_class = y_pred[..., class_idx]
        
        y_true_f = tf.reshape(y_true_class, [-1])
        y_pred_f = tf.reshape(y_pred_class, [-1])
        
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
        
        return (2. * intersection + smooth) / (union + smooth)
    
    dice.__name__ = f'dice_{class_name}'
    return dice


#compile model
mean_iou = tf.keras.metrics.MeanIoU(num_classes=CLASSES, name='mean_iou')

model.compile(optimizer=Adam(1e-4),
                loss=combined_loss,
                metrics=[dice_coefficient_multiclass,
                        dice_coefficient_per_class(0, "background"),
                        dice_coefficient_per_class(1, "glioma"),
                        dice_coefficient_per_class(2, "meningioma"),
                        dice_coefficient_per_class(3, "pituitary"),
                        mean_iou
                        ]
            )


train_generator = LazySegmentationGenerator(
    'dataset_split_segmentation/train/images',
    'dataset_split_segmentation/train/masks',
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    input_size=INPUT_SIZE,
    augmentation=training_augmentation,
    shuffle=True,
    workers=8,
    use_multiprocessing=True
)

val_generator = LazySegmentationGenerator(
    'dataset_split_segmentation/val/images',
    'dataset_split_segmentation/val/masks',
    batch_size=BATCH_SIZE,
    classes=CLASSES,
    input_size=INPUT_SIZE,
    augmentation=validation_augmentation,
    shuffle=False,
    workers=8,
    use_multiprocessing=True
)

print(f"✓ Training batches per epoch: {len(train_generator)}")
print(f"✓ Validation batches per epoch: {len(val_generator)}")
print(f"✓ Memory usage: MINIMAL (lazy loading)")

callbacks = [
    ModelCheckpoint('tumor_segmentation_best.keras', monitor='val_loss',verbose=1, save_best_only=True, mode='min'),
    EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1, min_lr=1e-7),
    CSVLogger('training_tumor_segmentation.csv'),
    TensorBoard(log_dir='./logs/tensorboard')
]

print("Starting training...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

model.save('tumor_segmentation_final.keras')
