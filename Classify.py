import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json


train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)
 
train_generator = train_datagen.flow_from_directory(
    'data/train',
    batch_size=60,
    class_mode='binary',
    target_size=(224,224))
 
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)
 
validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    shuffle=True, #false
    batch_size = 20,
    class_mode='binary',
    target_size=(224,224))


conv_base = ResNet50(include_top=False,
                     weights='imagenet')
 
for layer in conv_base.layers:
    layer.trainable = False
 
x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
#x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)
 
 
 
optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

#steps_per_epoch = int( np.ceil(x_train.shape[0] / batch_size) )



#plt.ion() #enable for LossHistory
 
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.valAcc = []
        self.iteration = 0

    def on_epoch_end(self, batch, logs={}):
		
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.valAcc.append(logs.get('val_acc'))
        self.iteration+=1
 #       print(self.iteration)
  #      print(self.acc)
   #     print(self.valAcc)
        plt.plot(range(len(self.acc)), self.acc, '-bo', range(len(self.acc)), self.valAcc, '-rx')
        
        plt.show()
        plt.pause(0.001)
        

Callbacks = [
				#keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=0, mode='auto', baseline=0.8, restore_best_weights=False), 
				keras.callbacks.ModelCheckpoint(filepath='./best_model_weights.h5', save_weights_only = True, monitor='val_acc', mode='max',  save_best_only=True), 
				LossHistory()
			]


'''
### Let's load after 100 epcochs
model = load_model('models_100epochs/keras/model.h5')


history = model.fit_generator(
    generator=train_generator,
    epochs=100,
    validation_data=validation_generator,
    steps_per_epoch=20, #20
    validation_steps = 10, #10
    callbacks=Callbacks)







# architecture and weights to HDF5
model.save('models/keras/model.h5')

# architecture to JSON, weights to HDF5
model.save_weights('models/keras/weights.h5')
with open('models/keras/architecture.json', 'w') as f:
    f.write(model.to_json())


# architecture and weights from HDF5
model = load_model('models/keras/model.h5')


 
# architecture from JSON, weights from HDF5
with open('models/keras/architecture.json') as f:
    model = model_from_json(f.read())
model.load_weights('models/keras/weights.h5')
'''



with open('models_100epochs/keras/architecture.json') as f:
	model = model_from_json(f.read())

model.load_weights('best_model_weights.h5')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


print("Validation set accuracy: {:.0f}%".format(model.evaluate_generator(validation_generator, steps = 10 )[1] * 100))

validation_img_paths = ["data/validation/clio/99.jpg",
                        "data/validation/astra/88.jpg",
                        "data/validation/astra/33.jpg",
                        "data/validation/clio/44.jpg",
                        "data/validation/clio/55.jpg",
                        "data/validation/clio/66.jpg",
                        "data/validation/clio/77.jpg",
                        "data/validation/astra/66.jpg"]
img_list = [Image.open(img_path) for img_path in validation_img_paths]
#img_size, img_size
validation_batch = np.stack([preprocess_input(np.array(img.resize((224, 224)))) for img in img_list])
 
pred_probs = model.predict(validation_batch)
#print(pred_probs)
fig, axs = plt.subplots(nrows = 2, ncols = int(np.ceil(len(img_list)/2)), figsize=(400, 400))
for i, img in enumerate(img_list):
    ax = axs[i%2, i//2 ]
    ax.axis('off')
    ax.set_title("{:.0f}% Astra, {:.0f}% Clio".format(100*pred_probs[i,0],
                                                          100*pred_probs[i,1]))
    ax.imshow(img)


plt.show()
