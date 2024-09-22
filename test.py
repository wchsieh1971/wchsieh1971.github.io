from keras import layers, models

def create_alexnet(input_shape=(227, 227, 3), num_classes=1000):
    model = models.Sequential()
    
    # 第一層卷積
    model.add(layers.Conv2D(64, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # 第二層卷積
    model.add(layers.Conv2D(192, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    
    # 第三層卷積
    model.add(layers.Conv2D(384, kernel_size=(3, 3), padding='same', activation='relu'))
    
    # 第四層卷積
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    
    # 第五層卷積
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 全連接層
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# 創建模型
alexnet_model = create_alexnet(input_shape=(227, 227, 3), num_classes=1000)
alexnet_model.summary()
