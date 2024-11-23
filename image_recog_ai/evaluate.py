import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, validation_dir, img_size, batch_size):
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation accuracy: {accuracy * 100:.2f}%')

evaluate_model('strawberry_species_model.h5', 'path/to/validation_dataset', (128, 128), 32)