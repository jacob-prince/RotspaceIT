#!/bin/sh

sbatch run_lesioning_experiment.sh Faces sledgehammer conv1 False
sbatch run_lesioning_experiment.sh Scenes sledgehammer conv1 False

sbatch run_lesioning_experiment.sh Faces sledgehammer conv1 True
sbatch run_lesioning_experiment.sh Scenes sledgehammer conv1 True

sbatch run_lesioning_experiment.sh Scenes cascade-forward conv1 False

sbatch run_lesioning_experiment.sh Faces cascade-backward conv1 False
sbatch run_lesioning_experiment.sh Faces cascade-backward relu1 False
sbatch run_lesioning_experiment.sh Faces cascade-backward maxpool1 False
sbatch run_lesioning_experiment.sh Faces cascade-backward conv2 False
sbatch run_lesioning_experiment.sh Faces cascade-backward relu2 False
sbatch run_lesioning_experiment.sh Faces cascade-backward maxpool2 False
sbatch run_lesioning_experiment.sh Faces cascade-backward conv3 False
sbatch run_lesioning_experiment.sh Faces cascade-backward relu3 False
sbatch run_lesioning_experiment.sh Faces cascade-backward conv4 False
sbatch run_lesioning_experiment.sh Faces cascade-backward relu4 False
sbatch run_lesioning_experiment.sh Faces cascade-backward conv5 False
sbatch run_lesioning_experiment.sh Faces cascade-backward relu5 False
sbatch run_lesioning_experiment.sh Faces cascade-backward maxpool5 False
sbatch run_lesioning_experiment.sh Faces cascade-backward avgpool5 False
sbatch run_lesioning_experiment.sh Faces cascade-backward drop5 False
sbatch run_lesioning_experiment.sh Faces cascade-backward fc6 False
sbatch run_lesioning_experiment.sh Faces cascade-backward relu6 False
sbatch run_lesioning_experiment.sh Faces cascade-backward drop6 False
sbatch run_lesioning_experiment.sh Faces cascade-backward fc7 False
sbatch run_lesioning_experiment.sh Faces cascade-backward relu7 False
sbatch run_lesioning_experiment.sh Faces cascade-backward fc8 False

sbatch run_lesioning_experiment.sh Scenes cascade-backward conv1 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward relu1 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward maxpool1 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward conv2 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward relu2 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward maxpool2 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward conv3 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward relu3 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward conv4 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward relu4 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward conv5 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward relu5 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward maxpool5 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward avgpool5 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward drop5 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward fc6 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward relu6 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward drop6 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward fc7 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward relu7 False
sbatch run_lesioning_experiment.sh Scenes cascade-backward fc8 False

# sbatch run_lesioning_experiment.sh Faces single-layer conv1 False
# sbatch run_lesioning_experiment.sh Faces single-layer relu1 False
# sbatch run_lesioning_experiment.sh Faces single-layer maxpool1 False
# sbatch run_lesioning_experiment.sh Faces single-layer conv2 False
# sbatch run_lesioning_experiment.sh Faces single-layer relu2 False
# sbatch run_lesioning_experiment.sh Faces single-layer maxpool2 False
# sbatch run_lesioning_experiment.sh Faces single-layer conv3 False
# sbatch run_lesioning_experiment.sh Faces single-layer relu3 False
# sbatch run_lesioning_experiment.sh Faces single-layer conv4 False
# sbatch run_lesioning_experiment.sh Faces single-layer relu4 False
# sbatch run_lesioning_experiment.sh Faces single-layer conv5 False
# sbatch run_lesioning_experiment.sh Faces single-layer relu5 False
# sbatch run_lesioning_experiment.sh Faces single-layer maxpool5 False
# sbatch run_lesioning_experiment.sh Faces single-layer avgpool5 False
# sbatch run_lesioning_experiment.sh Faces single-layer drop6 False
# sbatch run_lesioning_experiment.sh Faces single-layer fc6 False
# sbatch run_lesioning_experiment.sh Faces single-layer relu6 False
# sbatch run_lesioning_experiment.sh Faces single-layer drop7 False
# sbatch run_lesioning_experiment.sh Faces single-layer fc7 False
# sbatch run_lesioning_experiment.sh Faces single-layer relu7 False
# sbatch run_lesioning_experiment.sh Faces single-layer fc8 False