import os, shutil
import numpy as np

original_data_dir = 'images_cropped'

base_dir = 'data'

categories =['astra', 'clio']
sub_dir = ['train', 'validation']

if not os.path.exists(base_dir):
	os.mkdir(base_dir)
	print('Created directory: ', base_dir)
	
for dir_type in sub_dir:
	train_test_val_dir = os.path.join(base_dir, dir_type)
	
	if not os.path.exists(train_test_val_dir):
		os.mkdir(train_test_val_dir)
		
	for cat in categories:
		dir_cat = os.path.join(train_test_val_dir, cat)
		
		if not os.path.exists(dir_cat):
			print('Created dirs: ', dir_cat)
			os.mkdir(dir_cat)
			
directory_dict = {}

np.random.seed(12)
for cat in categories:
	list_of_images = np.array(os.listdir(os.path.join(original_data_dir, cat)))
	print("{}: {} files".format(cat, len(list_of_images)))
	indexes = dict()
	indexes['validation'] = sorted(np.random.choice(len(list_of_images), size = 100, replace = False))
	
	indexes['train'] = list(set(range(len(list_of_images))) - set(indexes['validation']))
	
	for phase in sub_dir:
		for i, fname in enumerate(list_of_images[indexes[phase]]):
			source = os.path.join(original_data_dir, cat, fname)
			destination = os.path.join(base_dir, phase, cat, str(i)+".jpg")
			shutil.copyfile(source,destination)
		print("{}, {}: {} files copied".format(cat, phase, len(indexes[phase])))
		directory_dict[phase + "_" + cat + "_dir"] = os.path.join(base_dir, phase, cat)
		
		
print(directory_dict)



#print('Total training Astra images:', len(os.listdir(directories_dict['train_alien_dir'])))
#print('Total training Clio images:', len(os.listdir(directories_dict['train_predator_dir'])))
#print("-"*32)
#print('Total validation Astra images:', len(os.listdir(directories_dict['validation_alien_dir'])))
#print('Total validation Clio images:', len(os.listdir(directories_dict['validation_ _dir'])))


