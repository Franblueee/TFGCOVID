
#from segment import *
from splitTrainTest import *
#from CIT import *

import os


if __name__ == "__main__":

    nombre = "COVIDGR1.0reducido"
    nombre = "COVIDGR1.0"

    image_dir = "data" + os.sep + "input" + os.sep + nombre + "-SinSegmentar"
    cropped_dir = "data" + os.sep + "generated" + os.sep + nombre + "-cropped"
    cropped_split_dir = "data" + os.sep + "generated" + os.sep + nombre + "-cropped-split"
    transformed_dir = "data" + os.sep + "generated" + os.sep + nombre + "-transformed-split"

    SEED = 31416

    #crop(image_dir, cropped_dir)
    #splitTrainTest(SEED, cropped_dir, cropped_split_dir, 0.8, 0.1)
    #transform()
    #splitInFolders()
    #makePartitions()
    #train()