import os
import argparse
import shutil
import numpy as np


# construct the argument parse and parse the arguments




original_data_dir = 'images'

base_dir = 'images_cropped'
categories =['astra', 'clio']



if not os.path.exists(base_dir):
	os.mkdir(base_dir)
	print('Created directory: ', base_dir)
	
for dir_type in categories:
	
	subdir = os.path.join(base_dir, dir_type)
	
	if not os.path.exists(subdir):
		os.mkdir(subdir)
					
directory_dict = {}

np.random.seed(12)
for cat in categories:
	list_of_images = np.array(os.listdir(os.path.join(original_data_dir, cat)))
	print("{}: {} files".format(cat, len(list_of_images)))

	for fname in list_of_images:
		
		source = os.path.join(original_data_dir, cat, fname)
		destination = os.path.join(base_dir, cat, fname)
		
		print("source file: {}".format(source))
		print("destination: {}".format(destination))
		Command2Exec = "python /home/andrzej/Python/PyImage/2-ImageDetection/DL_object_detection.py -i {} -o {}".format(source, destination)
		os.system(Command2Exec)

