from data_preprocessing import create_data_generators
from model import build_model

img_size = (128, 128)
batch_size = 32
train_dir = 'path/to/train_dataset'
validation_dir = 'path/to/validation_dataset'

train_generator, validation_generator = create_data_generators(train_dir, validation_dir, img_size, batch_size)
model = build_model((128, 128, 3), len(train_generator.class_indices))
model.fit(train_generator, epochs=20)
model.save('strawberry_species_model.h5')
