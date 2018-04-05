try:
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.python.keras.layers import BatchNormalization
except:
    from tensorflow.contrib.keras.python.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.contrib.keras.python.keras.models import Sequential
    from tensorflow.contrib.keras.python.keras.layers.normalization import BatchNormalization
from utils import Squeeze

model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(None, None, 1)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(4, 4), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(200, kernel_size=(4, 4), activation='relu'),
    Conv2D(200, kernel_size=(1, 1), activation='relu'),
    Conv2D(3, kernel_size=(1, 1), activation='softmax'),  # filters num == # of labels
    Squeeze(3),
])