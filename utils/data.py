
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import os
from PIL import Image


def filterDataset(folder, mode='train', label_map=None, newTrainClasses=None):
  annFile = '{}/annotations/instances_{}2017.json'.format(folder, mode)
  coco = COCO(annFile)
  images = []
  classes = []
  for id in coco.imgs:
    ann_ids = coco.getAnnIds(imgIds=[id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    if(len(anns) == 1) : 
        if(newTrainClasses != None and anns[0]['category_id'] not in newTrainClasses):
            continue
        if(label_map != None and anns[0]['category_id'] not in label_map):
            continue
        images+=coco.loadImgs(id)
        classes.append(anns[0]['category_id'])
  dataset_size = len(images)
  return images, dataset_size, classes, coco

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img

def getNormalMask(imageObj, coco, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'],  iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats([anns[0]['category_id']])
    train_mask = np.zeros(input_image_size)
    className = cats[0]['name']
    pixel_value = anns[0]['category_id']
    new_mask = cv2.resize(coco.annToMask(anns[0])*pixel_value, input_image_size)
    train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask 

def getBinaryMask(imageObj, coco, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask

def getClassId(imageObj, coco):
    annIds = coco.getAnnIds(imageObj['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    return anns[0]['category_id']

def dataGeneratorCoco(images, coco, folder, 
                      input_image_size=(224,224), batch_size=4, mode='train', mask_type='binary'):
  img_folder = '{}/{}2017'.format(folder, mode)
  dataset_size = len(images)
  c = 0
  while(True):
    img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
    mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')
    for i in range(c, c+batch_size): 
      imageObj = images[i]

      train_img = getImage(imageObj, img_folder, input_image_size)
      ### Create Mask ###
      if mask_type=="binary":
          train_mask = getBinaryMask(imageObj, coco, input_image_size)
      
      elif mask_type=="normal":
          train_mask = getNormalMask(imageObj, coco, input_image_size)
      img[i-c] = train_img
      mask[i-c] = train_mask
    classId = getClassId(imageObj, coco)
    c+=batch_size
    if(c + batch_size >= dataset_size):
        c=0
        random.shuffle(images)
    yield img, mask, classId
    # yield img, mask

def visualizeGenerator(gen):
    img, mask, classId = next(gen)
    
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
    
    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                        subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if(i==1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:,:,0])
                
            ax.axis('off')
            fig.add_subplot(ax)            
    plt.show()


def getLabelMap(newTrainClasses=None):
    val_dataset = COCODataset(folder='dataset', mode='val')
    val_classes = []
    for i, (img, mask, classId) in enumerate(val_dataset):
        if classId in newTrainClasses:
            val_classes.append(classId)
    val_classes = set(val_classes)
    label_map = {}
    sorted_val_classes = sorted(list(val_classes))
    for i in range(len(sorted_val_classes)):
        label_map[sorted_val_classes[i]] = i + 1
    return label_map


def getNewTrainClasses(numSamples=500):
    train_dataset = COCODataset(folder='dataset', mode='train')
    train_counts = {}
    for i in range(1000):
        train_counts[i] = 0
    for image, mask, classId in train_dataset:
        train_counts[classId] += 1

    newTrainClasses = []
    for k, v in train_counts.items():
        if v >= numSamples:
            newTrainClasses.append(k)
    return newTrainClasses


class COCODataset(Dataset):
    def __init__(self, folder, mode='train', input_image_size=(224,224), label_map=None, newTrainClasses=None):
        self.images, self.dataset_size, self.classes, self.coco = filterDataset(folder, mode, label_map, newTrainClasses)
        self.folder = folder
        self.input_image_size = input_image_size
        self.img_folder = '{}/{}2017'.format(folder, mode)
        self.label_map = label_map
        self.curr_len = self.dataset_size

    def __len__(self):
        return self.curr_len


    def __getitem__(self, idx):
        imageObj = self.images[idx]

        img = getImage(imageObj, self.img_folder, self.input_image_size)
        mask = getBinaryMask(imageObj, self.coco, self.input_image_size)
        
        classId = getClassId(imageObj, self.coco)
        if self.label_map != None and classId in self.label_map:
            classId = self.label_map[classId]  # map class labels to values from 0 to 7
        
        _image = np.array(img)
        image = torch.from_numpy(_image)
        image = image.permute(2, 0, 1)

        _mask = np.array(mask)
        _mask = torch.from_numpy(_mask)
        _mask = _mask.permute(2, 0, 1)
        _mask = _mask * classId

        return image, _mask, classId


class FacesDataset(Dataset):
    def __init__(self, folder='UTKFace', transform=None, age_filter=None):
        self.data_list = []
        self.transform = transform
        self.folder = folder
        
        for filename in os.listdir(self.folder):
            components = filename.split('_')
            if len(components) == 4:
                age = int(components[0])
                gender = int(components[1])
                ethnicity = int(components[2])
                img = Image.open(f'{self.folder}/{filename}')
                img = self.transform(img)
                img = img.type(torch.FloatTensor)

                self.data_list.append([img, age, gender, ethnicity])
            

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        return self.data_list[idx]


def data_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_metrics(gt_vec, pred_vec):
    accuracy = perf.accuracy_score(gt_vec, pred_vec)
    f1 = perf.f1_score(gt_vec, pred_vec, average='macro')
    
    return f1, accuracy