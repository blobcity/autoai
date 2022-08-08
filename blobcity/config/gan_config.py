# Copyright 2021 BlobCity, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense,Flatten, BatchNormalization,Activation, ZeroPadding2D,LeakyReLU,UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class DCGAN:
    
    def __init__(self,GENERATE_RES=3,IMAGE_CHANNELS=3,PREVIEW_ROWS = 4,PREVIEW_COLS = 7,PREVIEW_MARGIN = 16,SEED_SIZE = 100,
        DATA_PATH = "./",BATCH_SIZE = 32,BUFFER_SIZE = 60000,CROSS_ENTROPY = tf.keras.losses.BinaryCrossentropy()):

        self.GENERATE_RES = GENERATE_RES # Generation resolution factor 
        # (1=32, 2=64, 3=96, 4=128, etc.)
        self.GENERATE_SQUARE = 32 * self.GENERATE_RES # rows/cols (should be square)
        self.IMAGE_CHANNELS = IMAGE_CHANNELS

        # Preview image 
        self.PREVIEW_ROWS = PREVIEW_ROWS
        self.PREVIEW_COLS = PREVIEW_COLS
        self.PREVIEW_MARGIN = PREVIEW_MARGIN

        # Size vector to generate images from
        self.SEED_SIZE = SEED_SIZE

        # Configuration
        self.DATA_PATH = DATA_PATH
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.CROSS_ENTROPY = CROSS_ENTROPY

    def build_generator(self):
        GENERATE_RES=self.GENERATE_RES
        model = Sequential()

        model.add(Dense(4*4*256,activation="relu",input_dim=self.SEED_SIZE))
        model.add(Reshape((4,4,256)))

        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(UpSampling2D())
        model.add(Conv2D(256,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128,kernel_size=3,padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        if GENERATE_RES>1:
            model.add(UpSampling2D(size=(GENERATE_RES,GENERATE_RES)))
            model.add(Conv2D(128,kernel_size=3,padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))

        model.add(Conv2D(self.IMAGE_CHANNELS,kernel_size=3,padding="same"))
        model.add(Activation("tanh"))

        return model


    def build_discriminator(self,image_shape):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dropout(0.25))
        model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        return model

    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.CROSS_ENTROPY(tf.ones_like(real_output), real_output)
        fake_loss = self.CROSS_ENTROPY(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self,fake_output):
        return self.CROSS_ENTROPY(tf.ones_like(fake_output), fake_output)