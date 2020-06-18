from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Conv2D,Conv2DTranspose,Input,Cropping2D,add,Dropout,Reshape,Activation
from tensorflow.python.keras.utils import plot_model

def FCN32(nClasses,input_height,input_width):

    img_input = Input(shape=(input_height,input_width,3))
    model = vgg16.VGG16(include_top=False,weights='imagenet',input_tensor=img_input)
    # vgg去除全连接层为：7x7x512
    # vgg:5个block，1:filters：64，kernel：3；3-128；3-256；3-512
    o = Conv2D(filters=1024,kernel_size=(7,7),padding='same',activation='relu',name='fc6')(model.output)
    o = Dropout(0.5)(o)
    o = Conv2D(filters=1024,kernel_size=(1,1),padding='same',activation='relu',name='fc7')(o)
    o = Dropout(0.5)(o)

    o = Conv2D(filters=nClasses,kernel_size=(1,1),padding='same',activation='relu',name='score_fr')(o)
    o = Conv2DTranspose(filters=nClasses,kernel_size=(32,32),strides=(32,32),padding='valid',activation=None,name='score2')(o)
    o = Reshape((-1,nClasses))(o)
    o = Activation("softmax")(o)
    fcn8 = Model(img_input,o)
    return fcn8

if __name__=="__main__":
    model = FCN32(15,320,320)
    model.summary()
    # plot_model(model,show_shapes=True,to_file='fcn32.png')
