#!/bin/sh

# sbatch run_lesioning_experiment.sh Faces sledgehammer conv1 True
# sbatch run_lesioning_experiment.sh Scenes sledgehammer conv1 True

# sbatch run_lesioning_experiment.sh Faces sledgehammer conv1 True
# sbatch run_lesioning_experiment.sh Scenes sledgehammer conv1 True

# sbatch run_lesioning_experiment.sh Scenes cascade-backward conv2 True
# sbatch run_lesioning_experiment.sh Scenes cascade-backward relu2 True
# sbatch run_lesioning_experiment.sh Scenes cascade-backward maxpool2 True
# sbatch run_lesioning_experiment.sh Scenes cascade-backward conv3 True
# sbatch run_lesioning_experiment.sh Scenes cascade-backward relu3 True
#sbatch run_lesioning_experiment.sh Faces cascade-backward conv4 True
# sbatch run_lesioning_experiment.sh Scenes cascade-backward relu4 True
# sbatch run_lesioning_experiment.sh Scenes cascade-backward relu5 True
sbatch run_lesioning_experiment.sh Scenes cascade-backward drop5 True
sbatch run_lesioning_experiment.sh Scenes cascade-backward fc7 True

# sbatch run_lesioning_experiment.sh Faces cascade-forward conv1 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward relu1 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward maxpool1 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward conv2 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward relu2 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward maxpool2 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward conv3 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward relu3 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward conv4 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward relu4 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward conv5 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward relu5 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward maxpool5 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward avgpool5 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward drop5 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward fc6 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward relu6 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward drop6 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward fc7 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward relu7 True
# sbatch run_lesioning_experiment.sh Faces cascade-forward fc8 True

# sbatch run_lesioning_experiment.sh Scenes cascade-forward conv1 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward relu1 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward maxpool1 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward conv2 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward relu2 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward maxpool2 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward conv3 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward relu3 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward conv4 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward relu4 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward conv5 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward relu5 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward maxpool5 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward avgpool5 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward drop5 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward fc6 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward relu6 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward drop6 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward fc7 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward relu7 True
# sbatch run_lesioning_experiment.sh Scenes cascade-forward fc8 True

# # sbatch run_lesioning_experiment.sh Faces single-layer conv1 True
# # sbatch run_lesioning_experiment.sh Faces single-layer relu1 True
# # sbatch run_lesioning_experiment.sh Faces single-layer maxpool1 True
# # sbatch run_lesioning_experiment.sh Faces single-layer conv2 True
# # sbatch run_lesioning_experiment.sh Faces single-layer relu2 True
# # sbatch run_lesioning_experiment.sh Faces single-layer maxpool2 True
# # sbatch run_lesioning_experiment.sh Faces single-layer conv3 True
# # sbatch run_lesioning_experiment.sh Faces single-layer relu3 True
# # sbatch run_lesioning_experiment.sh Faces single-layer conv4 True
# # sbatch run_lesioning_experiment.sh Faces single-layer relu4 True
# # sbatch run_lesioning_experiment.sh Faces single-layer conv5 True
# # sbatch run_lesioning_experiment.sh Faces single-layer relu5 True
# # sbatch run_lesioning_experiment.sh Faces single-layer maxpool5 True
# # sbatch run_lesioning_experiment.sh Faces single-layer avgpool5 True
# # sbatch run_lesioning_experiment.sh Faces single-layer drop5 True
# # sbatch run_lesioning_experiment.sh Faces single-layer fc6 True
# # sbatch run_lesioning_experiment.sh Faces single-layer relu6 True
# # sbatch run_lesioning_experiment.sh Faces single-layer drop6 True
# # sbatch run_lesioning_experiment.sh Faces single-layer fc7 True
# # sbatch run_lesioning_experiment.sh Faces single-layer relu7 True
# # sbatch run_lesioning_experiment.sh Faces single-layer fc8 True

# sbatch run_lesioning_experiment.sh Scenes single-layer conv1 True
# sbatch run_lesioning_experiment.sh Scenes single-layer relu1 True
# sbatch run_lesioning_experiment.sh Scenes single-layer maxpool1 True
# sbatch run_lesioning_experiment.sh Scenes single-layer conv2 True
# # sbatch run_lesioning_experiment.sh Scenes single-layer relu2 True
# sbatch run_lesioning_experiment.sh Scenes single-layer maxpool2 True
# sbatch run_lesioning_experiment.sh Scenes single-layer conv3 True
# sbatch run_lesioning_experiment.sh Scenes single-layer relu3 True
# sbatch run_lesioning_experiment.sh Scenes single-layer conv4 True
# sbatch run_lesioning_experiment.sh Scenes single-layer relu4 True
# sbatch run_lesioning_experiment.sh Scenes single-layer conv5 True
# sbatch run_lesioning_experiment.sh Scenes single-layer relu5 True
# sbatch run_lesioning_experiment.sh Scenes single-layer maxpool5 True
# sbatch run_lesioning_experiment.sh Scenes single-layer avgpool5 True
# sbatch run_lesioning_experiment.sh Scenes single-layer drop5 True
# sbatch run_lesioning_experiment.sh Scenes single-layer fc6 True
# sbatch run_lesioning_experiment.sh Scenes single-layer relu6 True
# sbatch run_lesioning_experiment.sh Scenes single-layer drop6 True
# sbatch run_lesioning_experiment.sh Scenes single-layer fc7 True
# sbatch run_lesioning_experiment.sh Scenes single-layer relu7 True
# sbatch run_lesioning_experiment.sh Scenes single-layer fc8 True
