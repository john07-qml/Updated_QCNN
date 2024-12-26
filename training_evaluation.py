from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import QCNN_model

def train_qcnn_model(X_train, y_train, X_test, y_test, input_shape, epochs=1):
    model = QCNN_model.create_qcnn_model(input_shape)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Convert y_train and y_test back to one-hot encoding for QCNN training
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, 4)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, 4)

    X_train_expanded = tf.expand_dims(X_train, axis=-1)
    X_test_expanded = tf.expand_dims(X_test, axis=-1)

    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(
        X_train_expanded, y_train_one_hot,
        epochs=epochs,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

    loss, accuracy = model.evaluate(X_test_expanded, y_test_one_hot)
    print(f"Test accuracy: {accuracy}")

    model.save("qcnn_model.h5")
    print("Model saved")
    return model, history
