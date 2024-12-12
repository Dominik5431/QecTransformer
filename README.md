Code for the second half of my master's project. Training a transformer-based neural network to estimate logical operators from syndromes.

main.py includes all scripts for execution. That is, data generation, training, inference and plotting.

src contains the framework for the model, the datasets etc.
src/loops.py contains the training and inference loop.
src/nn defines the model structure.
src/error_code builds the circuit of the Surface code using stim.
src/data contains custom dataset classes for the project.

code contains the logical operators and stabilizers for the rotated surface code up to distance 13 and the steane code up to distance 5.
