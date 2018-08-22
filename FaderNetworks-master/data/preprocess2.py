

#!/usr/bin/env python
import os
import os.path
import matplotlib.image as mpimg
import cv2
import numpy as np
import torch
import csv

# preprocessing for the KDEF DB

ATTRIBUTES = "Neutral,Happy,Sad,Surprise,Fear,Disgust,Anger,Contempt"
ATTRIBUTES_FILE = "./attrfile2.txt"
ATTRIBUTES_CSV = 'training2.csv'
IMG_SIZE = 256
IMG_PATH = 'images_%i_%i_2.pth' % (IMG_SIZE, IMG_SIZE)
ATTR_PATH = 'attributes2.pth'
PATH_TO_IMAGES = 'KDEF/'
#N_IMAGES = 4900
N_IMAGES = 2940

images_names = [['NEHL.JPG', 'HAHL.JPG', 'SAHL.JPG', 'SUHL.JPG', 'AFHL.JPG', 'DIHL.JPG', 'ANHL.JPG'],
                ['NEHR.JPG', 'HAHR.JPG', 'SAHR.JPG', 'SUHR.JPG', 'AFHR.JPG', 'DIHR.JPG', 'ANHR.JPG'],
                ['NES.JPG',  'HAS.JPG',  'SAS.JPG',  'SUS.JPG',  'AFS.JPG',  'DIS.JPG',  'ANS.JPG' ]]


'''images_names = [['NEFL.JPG', 'HAFL.JPG', 'SAFL.JPG', 'SUFL.JPG', 'AFFL.JPG', 'DIFL.JPG', 'ANFL.JPG'],
     ['NEHL.JPG', 'HAHL.JPG', 'SAHL.JPG', 'SUHL.JPG', 'AFHL.JPG', 'DIHL.JPG', 'ANHL.JPG'],
     ['NEHR.JPG', 'HAHR.JPG', 'SAHR.JPG', 'SUHR.JPG', 'AFHR.JPG', 'DIHR.JPG', 'ANHR.JPG'],
     ['NEFR.JPG', 'HAFR.JPG', 'SAFR.JPG', 'SUFR.JPG', 'AFFR.JPG', 'DIFR.JPG', 'ANFR.JPG'],
     ['NES.JPG',  'HAS.JPG',  'SAS.JPG',  'SUS.JPG',  'AFS.JPG',  'DIS.JPG',  'ANS.JPG' ]]
     '''

sessions_id = ["A", "B"]
genders_id = ["M", "F"]

    
def create_attributes():
    
    attrFile = open(ATTRIBUTES_FILE, 'w+')
    #attrFile.write(str(N_IMAGES) + '\n')
    attrFile.write(ATTRIBUTES + '\n')
    for i in range (1,36):
        for session in sessions_id:
            for gender in genders_id:
                index = '%02d' % i
                dir_name = session + gender + index
                for angle in images_names:
                    for j, emotion in enumerate(angle):
                        file_name = dir_name + '/' + session + gender + index + emotion
                        outStr = file_name
                        for k in range(8):
                            outStr += '\t'
                            if k == j:
                                outStr += "1"
                            else:
                                outStr += "-1"
                        attrFile.write(outStr + '\n')
    attrFile.close()
    return

def preprocess_images():

    if os.path.isfile(IMG_PATH):
        print("%s exists, nothing to do." % IMG_PATH)
        return

    print("Reading images from " + PATH_TO_IMAGES + " ...")
    raw_images = []
    with open(ATTRIBUTES_FILE, "r") as attrFile:
        i = 0
        for line in attrFile:
            if i % 1000 == 0:
                print(i)
            if i >= 1:
                filePath = (line.split('\t'))[0]
                raw_images.append(mpimg.imread(PATH_TO_IMAGES + filePath))
                #raw_images.append(mpimg.imread('AffectNet/Manually_Annotated_Images/' + filePath)[20:-20])
            i += 1
            
    print("Resizing images ...")
    all_images = []
    for i, image in enumerate(raw_images):
        if i % 1000 == 0:
            print(i)
        #assert image.shape[0] == image.shape[1] and image.shape[2] == 3
        if IMG_SIZE < image.shape[0] and IMG_SIZE < image.shape[1]:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        #elif IMG_SIZE > image.shape[0]:
        else:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
        assert image.shape == (IMG_SIZE, IMG_SIZE, 3)
        all_images.append(image)

    data = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
    data = torch.from_numpy(data)
    print data.size()
    assert data.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(data[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(data, IMG_PATH)

def preprocess_attributes():

    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return

    attr_lines = [line.rstrip() for line in open(ATTRIBUTES_FILE, 'r')]
    assert len(attr_lines) == N_IMAGES + 1

    #attr_keys = attr_lines[1].split()
    attr_keys=["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in attr_keys}

    for i, line in enumerate(attr_lines[1:]):
        image_id = i + 1
        split = line.split()
        #assert len(split) == 41
        #assert split[0] == ('%06i.jpg' % image_id)
        assert all(x in ['-1', '1'] for x in split[1:])
        for j, value in enumerate(split[1:]):
            attributes[attr_keys[j]][i] = value == '1'

    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, ATTR_PATH)

create_attributes()
preprocess_images()
preprocess_attributes()
