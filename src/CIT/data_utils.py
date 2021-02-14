import os

from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage, Resize

def save_images(G_dict, input_img, label, class_names, path, name, classifier_name):
    for class_name in class_names:
        name = name.split(".")[0]
        tr_img = G_dict[class_name](input_img)
        tr_img = ToPILImage()(tr_img[0].cpu().detach())
        resnet_type = classifier_name.lower().split("t")[-1]
        print("Saving: " + path + "/"+"{}_{}T{}{}.png".format(name, class_names[label.item()],resnet_type, class_name))
        tr_img.save(path+"/"+"{}_{}T{}{}.png".format(name, class_names[label.item()],resnet_type,class_name))

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        name = os.path.basename(self.imgs[index][0])
        # name = str(self.imgs[index][0].split('/')[-1])
        #print(self.imgs[index][0])
        path = self.imgs[index][0].split('/')[-3]+os.sep+self.imgs[index][0].split('/')[-2]
        tuple_with_name = (original_tuple + (name,)+(path,))
        return tuple_with_name


class ImageFolderWithPaths_noUps(datasets.ImageFolder):
    """Custom dataset that includes image paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __init__(self, data_dir, data_size, img_transforms=None):
        super(ImageFolderWithPaths_noUps, self).__init__(data_dir, transform=img_transforms)
        self.data_size = data_size

    # override the __getitem__ method that dataloader calls
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths_noUps, self).__getitem__(index)

        hr_scale = Resize(size=(self.data_size, self.data_size), interpolation=Image.BICUBIC)
        width, height = original_tuple[0].size
        if width == self.data_size and height == self.data_size:
            hr_image = original_tuple[0]
        else:
            hr_image = hr_scale(original_tuple[0])

        return ToTensor()(hr_image), ToTensor()(hr_image), original_tuple[1]