import train
import evaluate

if __name__ == "__main__":
    train.train_model()  # Assuming you define a train_model function in train.py
    evaluate.evaluate_model('strawberry_species_model.h5', 'path/to/validation_dataset', (128, 128), 32)
