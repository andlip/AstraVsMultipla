# AstraVsMultipla
For testing transfer learning

General workflow:

- use JS commands from the browser
	+ navigate to google search -> Opel Astra -> images -> sroll down -> scroll even more -> execute JS commands
	+ output will be list of images urls in a file "urls.txt"

- use download_images.py to... download images
	+ -u <path to urls.txt file>
	+ -o <output path>
	+ files are downlowaded as 00000xxx.jpg to the outpu path

- use PrepareData.py to select and crop pictures containing cars
	+ assumes pictures are organised as: images/astra/xxx.jpg and images/clio/xxx.jpg 
