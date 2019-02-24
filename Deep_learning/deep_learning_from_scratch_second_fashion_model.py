import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras


img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
x, y = prep_data(fashion_data)


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D

# Create a `Sequential` model called `second_fashion_model`. Don't add layers yet.
second_fashion_model = Sequential()


# Add the first `Conv2D` layer to `second_fashion_model`. 
# It should have 12 filters, a kernel_size of 3 and the `relu` activation function. 
# The first layer always requires that you specify the `input_shape`.  
# We have saved the number of rows and columns to the variables `img_rows` and `img_cols` respectively, so the input shape in this case is `(img_rows, img_cols, 1)`.
second_fashion_model.add(Conv2D(18, kernel_size=3,
          activation='relu',
          input_shape=(img_rows, img_cols, 1)))


# 1. Add 2 more convolutional (`Conv2D layers`) with 20 filters each, 'relu' activation, and a kernel size of 3. Follow that with a `Flatten` layer, and then a `Dense` layer with 100 neurons. 
# 2. Add your prediction layer to `second_fashion_model`.  This is a `Dense` layer.  We alrady have a variable called `num_classes`.  Use this variable when specifying the number of nodes in this layer. The activation should be `softmax` (or you will have problems later).
second_fashion_model.add(Conv2D(24, activation='relu', kernel_size=3))
second_fashion_model.add(Conv2D(24, activation='relu', kernel_size=3))
second_fashion_model.add(Flatten())
second_fashion_model.add(Dense(128, activation='relu'))
second_fashion_model.add(Dense(10, activation='softmax'))


# Compile second_fashion_model with the `compile` method.  Specify the following arguments:
# 1. `loss = "categorical_crossentropy"`
# 2. `optimizer = 'adam'`
# 3. `metrics = ['accuracy']`
second_fashion_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


# Run the command `second_fashion_model.fit`. The arguments you will use are
# 1. The data used to fit the model. First comes the data holding the images, and second is the data with the class labels to be predicted. Look at the first code cell (which was supplied to you) where we called `prep_data` to find the variable names for these.
# 2. `batch_size = 100`
# 3. `epochs = 4`
# 4. `validation_split = 0.2`
second_fashion_model.fit(x, y,
          batch_size=128,
          epochs=5,
          validation_split = 0.2)