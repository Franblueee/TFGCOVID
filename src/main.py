
from segment import *
#from CIT import *

import os


if __name__ == "__main__":

    image_dir = "data" + os.sep + "input" + os.sep + "COVIDGR1.0-SinSegmentar"
    save_cropped_dir = "data" + os.sep + "generated" + os.sep + "COVIDGR1.0-cropped"

    crop(image_dir, save_cropped_dir)
    #splitTrainTest()
    #transform()
    #splitInFolders()
    #makePartitions()
    #train()