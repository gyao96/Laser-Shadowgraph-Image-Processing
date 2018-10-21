import numpy as np
practice = True

default_sample_step = 3

target_size = 128

distance_threshold = 25

step_size = [9, 12]

max_radius = 60

max_neigh = 9

distance_threshold = 15

overlapping1_threshold = 0.75

overlapping2_threshold = 0.75

l1_norm_threshold = 4

if practice:
    slope_lower = 0.45

    slope_upper = np.pi-slope_lower
else:
    slope_lower = 0.6
    
    slope_upper = np.pi-slope_lower