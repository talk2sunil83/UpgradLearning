class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = Sequential([
            Rescaling(1./255, input_shape=input_shape),
            Conv2D(filters=16, kernel_size=3, activation='relu'),
            Conv2D(filters=16, activation='relu', kernel_size=3),
            MaxPooling2D(pool_size=2),
            Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, default=0.25, step=0.05)),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            Conv2D(filters=hp.Choice('num_filters', values=[32, 64], default=64), activation='relu', kernel_size=3),
            MaxPooling2D(pool_size=2),
            Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, default=0.25, step=0.05)),
            Flatten(),
            Dense(units=hp.Int('units', min_value=32, max_value=512, step=32, default=128), activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid'], default='relu')),
            Dropout(rate=hp.Float('dropout_3', min_value=0.0, max_value=0.5, default=0.25, step=0.05)),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(optimizer=keras.optimizers.Adam(hp.Float(
            'learning_rate',
            min_value=1e-4,
            max_value=1e-2,
            sampling='LOG',
            default=1e-3)),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        return model


input_shape = (img_height, img_width, 3)
hypermodel = CNNHyperModel(input_shape, len(class_names))

HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2
OUTPUT_DIR = os.path.normpath('D:\\KerasTuner\\Random')


tuner_rs = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    seed=SEED,
    max_trials=MAX_TRIALS,
    project_name='skin_cancer',
    executions_per_trial=EXECUTION_PER_TRIAL, overwrite=True, directory=OUTPUT_DIR)
tuner_rs.search(augmented_train_ds, val_ds,  epochs=10, verbose=1)

best_model_rs = tuner_rs.get_best_models(num_models=1)[0]
val_accuracy_rs = best_model_rs.evaluate(val_ds)[1]
print('Random search Validation Accuracy: ', val_accuracy_rs)


def build_model(input_shape, num_classes):
    model = Sequential([
        #  Rescaling : to bring values between [0,1]
        Rescaling(1./255, input_shape=input_shape),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])
    return model


model = build_model(input_shape=(img_height, img_width, 3), num_classes=len(class_names))


model = Sequential([
    #  Rescaling : to bring values between [0,1]
    Rescaling(1./255, input_shape=input_shape),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    # Dropout(0.25),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, kernel_regularizer=l2(0.01), activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


def build_model(input_shape, num_classes):
    model = Sequential([
        #  Rescaling : to bring values between [0,1]
        Rescaling(1./255, input_shape=input_shape),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(128, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])
    return model


model = build_model(input_shape=input_shape, num_classes=len(class_names))
