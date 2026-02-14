# 8.2.3 Building the model
# Suppress warnings
import os, pathlib
from deep_learning_with_python.data_paths import get_data_root
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# Listing 8.7 Instantiating a small convnet for dogs vs cats classification
from tensorflow import keras
from tensorflow.keras import layers

# 8.2.2
new_base_dir = get_data_root() / "cats_vs_dogs_small"

#
# Listing 8.9 Using image_dataset_from_directory to read images
from tensorflow.keras.utils import image_dataset_from_directory

test_dataset = image_dataset_from_directory(
        new_base_dir / "test",
        image_size=(180, 180),
        batch_size=32)

#
# Listing 8.13 Evaluating the model on the test set
test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")

