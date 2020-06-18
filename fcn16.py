from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.models import Model, Sequential,load_model
from tensorflow.python.keras.layers import Conv2D,Conv2DTranspose,Input,Cropping2D,add,Dropout,Reshape,Activation,UpSampling2D
from tensorflow.python.keras.utils import plot_model
from fcn32 import FCN32

def FCN16(nClasses,input_height,input_width):

    img_input = Input(shape=(input_height,input_width,3))
    # model = vgg16.VGG16(include_top=False,weights='imagenet',input_tensor=img_input)
    # vgg去除全连接层为：7x7x512
    # vgg:5个block，1:filters：64，kernel：3；3-128；3-256；3-512
    model = FCN32(11, 320, 320)
    model.load_weights("model.h5")


    skip1 = Conv2DTranspose(512,kernel_size=(3,3),strides=(2,2),padding='same',kernel_initializer="he_normal",name="upsampling6")(model.get_layer("fc7").output)
    summed = add(inputs=[skip1,model.get_layer("block4_pool").output])
    up7 = UpSampling2D(size=(16,16),interpolation='bilinear',name='upsamping_7')(summed)
    o = Conv2D(nClasses,kernel_size=(3,3),activation='relu',padding='same',name='conv_7')(up7)


    o = Reshape((-1,nClasses))(o)
    o = Activation("softmax")(o)
    fcn16 = Model(model.input,o)
    return fcn16

if __name__=="__main__":
    model = FCN16(15,320,320)
    model.summary()
    plot_model(model,show_shapes=True,to_file='fcn16.png')
    # plot_model(model,show_shapes=True,to_file='fcn32.png')
