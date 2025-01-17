# PathTracker Data Generator - Psychophysics Experiment 2

This repository generates pathtracker samples as follows:

![PathTracker Sample 1](https://github.com/grassknoted/pathtracker-data/blob/main/positive_sample_gifs/1_sample_1/GIF.gif)
![PathTracker Sample 2](https://github.com/grassknoted/pathtracker-data/blob/main/positive_sample_gifs/1_sample_4/GIF.gif)
![PathTracker Sample 3](https://github.com/grassknoted/pathtracker-data/blob/main/positive_sample_gifs/1_sample_7/GIF.gif)


The file `cluster_store_individual_modular_spline_curve_generator_path_finder_motion_race_distractor_many_fixed_blob_128_permanent_markers_square_numpy.py` is the main plotting logic, while `spline_scaled_curve_generator_class.py` is the class for generating coordinates of the trajectories beforehand. 

Parameters can be set in `gen_stim.sh` file.
Negative instance code is the same, but segregated in `gen_negative` directory for readability and organization. Similar parameters need to be set in `gen_negative/gen_stim.sh` as well, only difference being `NEGATIVE_SAMPLE` needs to be set to `1`, and the paths changed to reflect the change in directory where the data (.npy files) needs to be generated. 

In `gen_stim.sh`, the parameter `outer_path` is where the files are generated, segregated in positive and negative directories (1 and 0 respectively). The parameter `path` is used to restart the program if it gets killed or stalled for some reason. In that case, `start_sample` can be used to start from a specific sample where the earlier run stopped.
