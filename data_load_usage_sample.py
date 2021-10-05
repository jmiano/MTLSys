from utils import data

## Get image set with only one class
train_images, train_dataset_size, train_classes, train_coco = data.filterDataset('dataset', 'train')
val_images, val_dataset_size, val_classes, val_coco = data.filterDataset('dataset', 'val')

batch_size = 4
input_image_size = (224,224)
mask_type = 'normal'

val_gen = data.dataGeneratorCoco(val_images, val_coco, '.',
                            input_image_size, batch_size, 'val', mask_type)
data.visualizeGenerator(val_gen)