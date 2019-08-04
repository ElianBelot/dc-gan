# ===============[ IMPORTS ]===============
from keras.layers import *
from keras.models import *
from keras.optimizers import RMSprop
from PIL import Image

import numpy as np
import os


# ===============[ HYPERPARAMETERS ]===============
BATCH_SIZE = 64
ITERATIONS = 4096
SAMPLES_AMOUNT = 8

DATA_PATH = 'data/flower_images.npy'
SAMPLES_PATH = 'samples/'
CHECKPOINTS_PATH = 'checkpoints/'

ITERATIONS_PER_CHECKPOINT = 50
ITERATIONS_PER_SAMPLE = 50

IMAGE_SHAPE = [64, 64, 3]
LATENT_SHAPE = 64

DISCRIMINATOR_HISTORY = []
GENERATOR_HISTORY = []


# ===============[ DATA PRE-PROCESSING ]===============
X = np.load(DATA_PATH)
X = X.astype('float32') / 255.0


# ===============[ GENERATOR MODEL ]===============
def create_generator():
    generator = Sequential()

    # SHAPE: 4x4x64
    generator.add(Dense(4 * 4 * 64, input_shape=[LATENT_SHAPE], activation='relu'))
    generator.add(Reshape([4, 4, 64]))

    # SHAPE: 4x4x64
    generator.add(UpSampling2D())
    generator.add(Conv2D(64, 3, padding='same', activation='relu'))
    generator.add(BatchNormalization(momentum=0.8))

    # SHAPE: 8x8x64
    generator.add(UpSampling2D())
    generator.add(Conv2D(32, 3, padding='same', activation='relu'))
    generator.add(BatchNormalization(momentum=0.8))

    # SHAPE: 16x16x32
    generator.add(UpSampling2D())
    generator.add(Conv2D(16, 3, padding='same', activation='relu'))
    generator.add(BatchNormalization(momentum=0.8))

    # SHAPE: 32x32x16
    generator.add(UpSampling2D())
    generator.add(Conv2D(8, 3, padding='same', activation='relu'))
    generator.add(BatchNormalization(momentum=0.8))

    # SHAPE: 64x64x3
    generator.add(Conv2D(3, 1, padding='same', activation='sigmoid'))

    return generator


# ===============[ DISCRIMINATOR MODEL ]===============
def create_discriminator():
    discriminator = Sequential()

    # SHAPE: 64x64x3
    discriminator.add(Conv2D(8, 3, input_shape=IMAGE_SHAPE, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(AveragePooling2D())

    # SHAPE: 32x32x8
    discriminator.add(Conv2D(16, 3, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(AveragePooling2D())

    # SHAPE: 16x16x16
    discriminator.add(Conv2D(32, 3, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(AveragePooling2D())

    # SHAPE: 8x8x32
    discriminator.add(Conv2D(64, 3, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(AveragePooling2D())

    # SHAPE: 4x4x64
    discriminator.add(Conv2D(64, 3, padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.25))
    discriminator.add(Flatten())

    # SHAPE: 1x1x1
    discriminator.add(Dense(1, activation='sigmoid'))

    return discriminator


# ===============[ CREATE MODELS ]===============
discriminator = create_discriminator()
generator = create_generator()


# ===============[ BUILD DISCRIMINATOR ]===============
def build_discriminator():
    for layer in discriminator.layers:
        layer.trainable = True

    for layer in generator.layers:
        layer.trainable = False

    real_image = Input(IMAGE_SHAPE)
    validity_real = discriminator(real_image)

    fake_image = generator(Input([LATENT_SHAPE]))
    validity_fake = discriminator(fake_image)

    discriminator_model = Model(inputs=[real_image, Input([LATENT_SHAPE])], outputs=[validity_real, validity_fake])
    discriminator_model.compile(optimizer=RMSprop(lr=0.0002), loss=['mean_squared_error', 'mean_squared_error'])

    return discriminator_model


# ===============[ BUILD GENERATOR ]===============
def build_generator():
    for layer in discriminator.layers:
        layer.trainable = False

    for layer in generator.layers:
        layer.trainable = True

    fake_image = generator(Input([LATENT_SHAPE]))
    validity = discriminator(fake_image)

    generator_model = Model(inputs=Input([LATENT_SHAPE]), outputs=validity)
    generator_model.compile(optimizer=RMSprop(lr=0.0002), loss='mean_squared_error')

    return generator_model


# ===============[ BUILD MODELS ]===============
discriminator_model = build_discriminator()
generator_model = build_generator()


# ===============[ TRAINING ]===============
def train():
    for i in range(ITERATIONS):

        # Generating data
        real_labels = np.ones([BATCH_SIZE, 1])
        fake_labels = np.zeros([BATCH_SIZE, 1])

        image_indices = np.random.randint(0, X.shape[0] - 1, [BATCH_SIZE])
        real_images = X[image_indices]

        # Discriminator training
        latent_vectors = np.random.normal(0., 1., [BATCH_SIZE, 64])
        discriminator_loss = discriminator_model.train_on_batch([real_images, latent_vectors], [real_labels, fake_labels])
        DISCRIMINATOR_HISTORY.append(discriminator_loss[1] / 2 + discriminator_loss[2] / 2)

        # Generator training
        latent_vectors = np.random.normal(0., 1., [BATCH_SIZE, 64])
        generator_loss = generator_model.train_on_batch(latent_vectors, real_labels)
        GENERATOR_HISTORY.append(generator_loss)

        # Checkpoints
        if i % ITERATIONS_PER_CHECKPOINT == 0:
            print('\n\n' + f'[ ITERATION N°{i} ]'.center(50, '='))
            print(f' \t > Discriminator loss: \t {DISCRIMINATOR_HISTORY[-1]:.4f}')
            print(f' \t > Generator loss: \t {GENERATOR_HISTORY[-1]:.4f}')

            discriminator_model.save_weights(os.path.join(CHECKPOINTS_PATH, f'discriminator/discriminator-{i:02d}-{DISCRIMINATOR_HISTORY[-1]:.2f}.hdf5'))
            generator_model.save_weights(os.path.join(CHECKPOINTS_PATH, f'generator/generator-{i:02d}-{GENERATOR_HISTORY[-1]:.2f}.hdf5'))

        # Sample generation
        if i % ITERATIONS_PER_SAMPLE == 0:
            print(f' \t > Saved {SAMPLES_AMOUNT} image samples.')
            save_samples(iteration=i)


# ===============[ GENERATE SAMPLES ]===============
def save_samples(iteration):
    latent_vectors = np.random.normal(0., 1., [SAMPLES_AMOUNT, LATENT_SHAPE])
    fake_images = generator.predict(latent_vectors)

    for i in range(SAMPLES_AMOUNT):
        generated_image = Image.fromarray(np.uint8(fake_images[i] * 255))
        generated_image.save(os.path.join(SAMPLES_PATH, f'iteration-{iteration:02d}-{i+1:02d}.png'))


# ===============[ EXECUTION ]===============
train()
