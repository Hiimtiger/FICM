from utils.train import train_model
from inference import test_model_all

#================================

#train_model(train_dir = "TRAINING_SET/fluorescence", model_name="fluorescence_CNN", epochs=10, batch_size=4, lr=0.001)

#train_model(train_dir = "TRAINING_SET/grid", model_name="grid_CNN", epochs=10, batch_size=4, lr=0.001)

#train_model(train_dir = "TRAINING_SET/ficm", model_name="ficm_CNN", epochs=10, batch_size=4, lr=0.001)

#================================

test_model_all("TEST_SET/fluorescence", "utils/saved_model/fluorescence_CNN")
test_model_all("TEST_SET/grid", "utils/saved_model/grid_CNN")
test_model_all("TEST_SET/stacked", "utils/saved_model/stacked_CNN")
test_model_all("TEST_SET/ficm", "utils/saved_model/ficm_CNN")