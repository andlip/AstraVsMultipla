import os, shutil
import numpy as np
from PIL import Image


categories = ["train", "validation"]
cars = ["astra", "clio"]
base_dir = "data"


for cat in categories:
	for car in cars:
		dir_cat = os.path.join(base_dir, cat, car)
		list_of_images = np.array(os.listdir(dir_cat))
		
		for img in list_of_images:
			try:
				Image.open("{}/{}".format(dir_cat, img))
			except:
				os.remove("{}/{}".format(dir_cat, img))
				print("Removed file {}/{}".format(dir_cat,img))
