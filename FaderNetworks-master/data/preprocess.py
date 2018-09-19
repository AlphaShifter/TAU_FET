#!/usr/bin/env python
import os
import os.path
import matplotlib.image as mpimg
import cv2
import numpy as np
import torch
import csv

ATTRIBUTES = "Neutral,Happy,Sad,Surprise,Fear,Disgust,Anger,Contempt"
ATTRIBUTES_FILE = "./attrfile.txt"
ATTRIBUTES_CSV = 'training.csv'
IMG_SIZE = 256
IMG_PATH = 'images_%i_%i.pth' % (IMG_SIZE, IMG_SIZE)
ATTR_PATH = 'attributes.pth'
PATH_TO_IMAGES = 'Manually_Annotated_Images/'

def create_attributes():
    
    attrFile = open(ATTRIBUTES_FILE, 'w+')
    #attrFile.write(str(N_IMAGES) + '\n')
    attrFile.write(ATTRIBUTES + '\n')
    attr_count = [0,0,0,0,0,0,0,0]
    with open(ATTRIBUTES_CSV, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        count = 0
        for row in spamreader:
            rowlst = row[0].split(',')
            outStr = rowlst[0]
            
            ignoreRow = True
            if not os.path.isfile(PATH_TO_IMAGES+rowlst[0]):
                #print('AffectNet/Manually_Annotated_Images/'+rowlst[0])
                continue
            for i in range(8):
                outStr += '\t'
                if str(i) == rowlst[6]:
                    outStr += "1"
                    ignoreRow = False
                    count += 1
                    attr_count[i] += 1
                else:
                    outStr += "-1"
            if not ignoreRow:
                attrFile.write(outStr + '\n')
            #if(count>14000):
            #    break
        print attr_count
    attrFile.close()
    return count

def preprocess_images(N_IMAGES):

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
    assert data.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(data[:20000].clone(), 'images_%i_%i_20000.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(data, IMG_PATH)

def preprocess_attributes(N_IMAGES):

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

N_IMAGES = create_attributes()
preprocess_images(N_IMAGES)
preprocess_attributes(N_IMAGES)
