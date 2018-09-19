# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
from torch.autograd import Variable
from logging import getLogger


logger = getLogger()


AVAILABLE_ATTR = [
    "Neutral","Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"
]

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/')


def log_attributes_stats(train_attributes, valid_attributes, test_attributes, params):
    """
    Log attributes distributions.
    """
    k = 0
    for (attr_name, n_cat) in params.attr:
        logger.debug('Train %s: %s' % (attr_name, ' / '.join(['%.5f' % train_attributes[:, k + i].mean() for i in range(n_cat)])))
        logger.debug('Valid %s: %s' % (attr_name, ' / '.join(['%.5f' % valid_attributes[:, k + i].mean() for i in range(n_cat)])))
        logger.debug('Test  %s: %s' % (attr_name, ' / '.join(['%.5f' % test_attributes[:, k + i].mean() for i in range(n_cat)])))
        assert train_attributes[:, k:k + n_cat].sum() == train_attributes.size(0)
        assert valid_attributes[:, k:k + n_cat].sum() == valid_attributes.size(0)
        assert test_attributes[:, k:k + n_cat].sum() == test_attributes.size(0)
        k += n_cat
    assert k == params.n_attr


def load_images(params):
    """
    Load training datasets.
    """
    # load data
    images_filename = 'images_%i_%i_20000.pth' if params.debug else 'images_%i_%i.pth'
    images_filename = images_filename % (params.img_sz, params.img_sz)
    images2_filename = 'images_%i_%i_2.pth'
    images2_filename = images2_filename % (params.img_sz, params.img_sz)
    images = torch.load(os.path.join(DATA_PATH, images_filename))
    #images2 = torch.load(os.path.join(DATA_PATH, images2_filename))
    attributes = torch.load(os.path.join(DATA_PATH, 'attributes.pth'))
    #attributes2 = torch.load(os.path.join(DATA_PATH, 'attributes2.pth'))

    # parse attributes
    attrs = []
    #attrs2 = []
    for name, n_cat in params.attr:
        for i in range(n_cat):
            attrs.append(torch.FloatTensor((attributes[name] == i).astype(np.float32)))
            #attrs2.append(torch.FloatTensor((attributes2[name] == i).astype(np.float32)))
    attributes = torch.cat([x.unsqueeze(1) for x in attrs], 1)
    #attributes2 = torch.cat([x.unsqueeze(1) for x in attrs2], 1)
    
    # split train / valid / test
    if params.debug:
        train_index = 10000
        valid_index = 15000
        test_index = 20000
    else:
        train_index = int(float(len(images))*0.91)
        valid_index = int(float(len(images))*0.96)
        test_index = len(images)
    train_images = images[:train_index]
    valid_images = images[train_index:valid_index]
    test_images = images[valid_index:test_index]
    train_attributes = attributes[:train_index]
    valid_attributes = attributes[train_index:valid_index]
    test_attributes = attributes[valid_index:test_index]

    # log dataset statistics / return dataset
    logger.info('%i / %i / %i images with attributes for train / valid / test sets'
                % (len(train_images), len(valid_images), len(test_images)))
    log_attributes_stats(train_attributes, valid_attributes, test_attributes, params)
    images = train_images, valid_images, test_images
    attributes = train_attributes, valid_attributes, test_attributes

    return images, attributes, None, None#, images2, attributes2 

def normalize_images(images):
    """
    Normalize image values.
    """
    return images.float().div_(255.0).mul_(2.0).add_(-1)


class DataSampler(object):

    def __init__(self, images, attributes, images2, attributes2, params):
    #def __init__(self, images, attributes, params):

        """
        Initialize the data sampler with training data.
        """
        assert images.size(0) == attributes.size(0), (images.size(), attributes.size())
        self.images = images
        self.attributes = attributes
        self.images2 = images2
        self.attributes2 = attributes2
        self.batch_size = params.batch_size
        self.v_flip = params.v_flip
        self.h_flip = params.h_flip

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.images.size(0)

    def train_batch(self, bs):
        """
        Get a batch of random images with their attributes.
        """
        # image IDs
        idx = torch.LongTensor(2 * bs).random_(len(self.images))

        # select images / attributes
        batch_x1 = normalize_images(self.images.index_select(0, idx[0:bs-1]).cuda())
        batch_y1 = self.attributes.index_select(0, idx[0:bs-1]).cuda()
        batch_x2 = normalize_images(self.images.index_select(0, idx[bs:2*bs - 1]).cuda())
        batch_y2 = self.attributes.index_select(0, idx[bs:2*bs - 1]).cuda()
        
        # data augmentation
        if self.v_flip and np.random.rand() <= 0.5:
            batch_x1 = batch_x1.index_select(2, torch.arange(batch_x1.size(2) - 1, -1, -1).long().cuda())
            batch_x2 = batch_x2.index_select(2, torch.arange(batch_x2.size(2) - 1, -1, -1).long().cuda())
        if self.h_flip and np.random.rand() <= 0.5:
            batch_x1 = batch_x1.index_select(3, torch.arange(batch_x1.size(3) - 1, -1, -1).long().cuda())
            batch_x2 = batch_x2.index_select(3, torch.arange(batch_x2.size(3) - 1, -1, -1).long().cuda())

        return Variable(batch_x1, volatile=False), Variable(batch_y1, volatile=False), Variable(batch_x2, volatile=False), Variable(batch_y2, volatile=False)

    def eval_batch(self, i, j):
        """
        Get a batch of images in a range with their attributes.
        """
        assert i < j
        batch_x = normalize_images(self.images[i:j].cuda())
        batch_y = self.attributes[i:j].cuda()
        return Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
    
    def eval_batch2(self, i, j):
        """
        Get a batch of images in a range with their attributes.
        """
        assert i < j
        batch_x = normalize_images(self.images2[i:j].cuda())
        batch_y = self.attributes2[i:j].cuda()
        return Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
