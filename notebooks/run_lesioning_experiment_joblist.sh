#!/bin/sh

# sbatch run_lesioning_experiment.sh Faces sledgehammer conv1 False
# sbatch run_lesioning_experiment.sh Scenes sledgehammer conv1 False

sbatch run_lesioning_experiment.sh Faces sledgehammer conv1 True
sbatch run_lesioning_experiment.sh Scenes sledgehammer conv1 True

# sbatch run_lesioning_experiment.sh Faces cascade-forward maxpool5 False
# sbatch run_lesioning_experiment.sh Scenes cascade-forward maxpool5 False

# sbatch run_lesioning_experiment.sh Faces cascade-backward maxpool5 False
# sbatch run_lesioning_experiment.sh Scenes cascade-backward maxpool5 False

sbatch run_lesioning_experiment.sh Scenes cascade-forward conv1 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward relu1 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward maxpool1 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward conv2 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward relu2 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward maxpool2 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward conv3 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward relu3 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward conv4 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward relu4 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward conv5 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward relu5 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward maxpool5 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward avgpool5 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward drop5 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward fc6 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward relu6 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward drop6 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward fc7 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward relu7 False
sbatch run_lesioning_experiment.sh Scenes cascade-forward fc8 False

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