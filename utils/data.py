from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import Dataset

def filterDataset(folder, mode='train'):
  annFile = '{}/annotations/instances_{}2017.json'.format(folder, mode)
  coco = COCO(annFile)
  images = []
  classes = []
  for id in coco.imgs:
    ann_ids = coco.getAnnIds(imgIds=[id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    if(len(anns) == 1) : 
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
    pixel_value = 1
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



class COCODataset(Dataset):
    def __init__(self, folder, mode='train', input_image_size=(224,224)):
        self.images, self.dataset_size, self.classes, self.coco = filterDataset(folder, mode)
        self.folder = folder
        self.input_image_size = input_image_size
        self.img_folder = '{}/{}2017'.format(folder, mode)

    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        imageObj = self.images[idx]

        img = getImage(imageObj, self.img_folder, self.input_image_size)
        mask = getNormalMask(imageObj, self.coco, self.input_image_size)
        
        classId = getClassId(imageObj, self.coco)
        
        _image = np.array(img)
        image = torch.from_numpy(_image)
        image = image.permute(2, 0, 1)

        _mask = np.array(mask)
        _mask = torch.from_numpy(_mask)
        _mask = _mask.permute(2, 0, 1)

        return image, _mask, classId