#CONTROLGAN for coco dataset
1. Get the meta-data for coco in a zip file [ https://drive.google.com/uc?id=1GOEl9lxgSsWUWOXkZZrch08GgPADze7U]. Unzip it in the data folder.
2. Get the training dataset in a zip file [http://images.cocodataset.org/zips/train2014.zip]. Unzip it in the coco folder. Rename it to images.
3. Get the valid dataset in a zip file. [http://images.cocodataset.org/zips/val2014.zip]. Unzip it in the coco folder.
4. Move the valid dataset contents to the images folder using this piece of code.
#-----------------------------------------------------------------------------
import os
import shutil
  
source = '/content/ControlGAN/data/coco/val2014/'
destination = '/content/ControlGAN/data/coco/images/'
allfiles = os.listdir(source)
for f in allfiles:
    shutil.move(source + f, destination + f)
    
#-----------------------------------------------------------------------------------
5. Make a folder in DAMSMencoders. Name it 'coco'. Load the text & image encoders to the folder.
6. Copy the path & add it to code/cfg/DAMSM/coco.yml in TRAIN.NET_E, & in cfg/train_coco.yml in TRAIN.NET_E paths.
7. Run the script pretrain_DAMSM.py to resume training of the encoders.
