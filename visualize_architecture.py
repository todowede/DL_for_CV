# import libraries
from Image_Classification.nn import LeNet
from tensorflow.keras.utils import plot_model

# Initialize Lenet and write the network arch
# visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenel.png", show_shapes=True)
# plot by runing python3 visualize_architecture.py
