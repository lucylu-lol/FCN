from tensorflow.python.keras.callbacks import ModelCheckpoint,TensorBoard
from fcn32 import FCN32
from fcn16 import FCN16
from fcn8 import FCN8
from tensorflow.python.keras.optimizers import SGD
import tensorflow as tf
import math
from load_batchs import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

trainimgPath = "CamVid/train"
trainsegPath = "CamVid/trainannot"
BATCH_SIZE = 2
n_classes = 11

epochs = 100

input_height = 320
input_width = 320

valimgPath = "CamVid/val"
valsegPath = "CamVid/valannot"

key = ""

# datagen = imageSegmentationGenerator
# dataset = tf.data.Dataset.from_generator(datagen,(tf.float16,tf.float16),(tf.TensorShape([None,None,None,None]),tf.TensorShape([None,None,None])))
# dataset = dataset.shuffle(100)
# # dataset = dataset.batch(BATCH_SIZE)
#
# dataval = segSegmentationGenerator
# dataval = tf.data.Dataset.from_generator(dataval,(tf.float16,tf.float16),(tf.TensorShape([None,None,None,None]),tf.TensorShape([None,None,None])))
# dataval = dataset.shuffle(100)



# model = FCN32(n_classes,input_height,input_width)
# model = FCN16(n_classes,input_height,input_width)
model = FCN8(n_classes,input_height,input_width)

optimizer = SGD(lr=2e-4,momentum=0.99,decay=1e-6)
model.compile(loss="categorical_crossentropy",optimizer=optimizer,metrics=['acc'])
trainGenerator = imageSegmentationGenerator(trainimgPath,trainsegPath,BATCH_SIZE,n_classes,input_height,input_width)
testGenerator = imageSegmentationGenerator(valimgPath,valsegPath,BATCH_SIZE,n_classes,input_height,input_width)
import datetime
# filename ="test_{}.hdf5".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

checkpoint = ModelCheckpoint(filepath="model8.h5",monitor='acc',mode='auto',save_best_only='True',save_weights_only=True)
# H = model.fit(dataset,epochs=epochs,callbacks=[checkpoint],validation_data=dataval)
H = model.fit_generator(trainGenerator,epochs=epochs,steps_per_epoch=367./BATCH_SIZE,callbacks=[checkpoint],validation_data=testGenerator,validation_steps=2)
Epochs = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,Epochs),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,Epochs),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,Epochs),H.history["acc"],label="train_acc")
plt.plot(np.arange(0,Epochs),H.history["val_acc"],label="val_acc")
plt.title("training loss and acc")
plt.xlabel("epochs")
plt.ylabel("loss/acc")
plt.legend()
plt.savefig("auc8.png")
plt.show()










