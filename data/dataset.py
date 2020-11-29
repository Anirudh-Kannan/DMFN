import torch.utils.data as data
from PIL import Image
import data.util as util
import random
import torchvision.transforms as transforms
import numpy as np
import torch
Rx = Ry = 1

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


# opt
# image_list | mask_flist | fineSize | mask_type | mask_pos | vertical_margin | horizontal_margin | max_delta_height | max_delta_width | center_crop

class ImageFilelist(data.Dataset):
    def __init__(self, opt, flist_reader=default_flist_reader, loader=default_loader):
        self.imlist = flist_reader(opt['image_list'])
        self.loader = loader
        self.opt = opt

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # [0, 1] --> [-1, 1]
        self.transform = transforms.Compose(transform_list)
        if opt['mask_list'] is not None and opt['mask_type'] == 'irregular':
            self.mask_data = flist_reader(opt['mask_list'])

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(impath)
        img = self.resize(img, self.opt['img_shape'][2], self.opt['img_shape'][1])
        img_tensor = self.transform(img)  # Tensor [C, H, W], [-1, 1]
        if self.opt['mask_type'] == 'regular':
            bbox_tensor, mask_tensor = self.load_mask(index, impath)  # Tensor [1, H, W]
        else:
            mask_tensor = self.load_mask(index)
            bbox = util.bbox(self.opt)
            bbox_tensor = torch.from_numpy(np.array(bbox))

        # generate mask, 1 represents masked point
        # mask: mask region {0, 1}
        # x_incomplete: incomplete image, [-1, 1]
        # returns: [-1, 1] as predicted image
        input_tensor = img_tensor * (1. - mask_tensor)  # [C, H, W]
        return {'input': input_tensor, 'bbox': bbox_tensor, 'mask': mask_tensor, 'target': img_tensor, 'paths': impath}

    def __len__(self):
        return len(self.imlist)

    def load_mask(self, index, impath):

        if self.opt['mask_type'] == 'regular':
            bbox = util.bbox(self.opt)
            x = impath.split('/')[-1].split('.')[0]

            flag = 0

            for i in range(1,6):
                file1 = open('/content/drive/My Drive/VCL/Bounding Boxes/finalboundingboxes'+ str(i) +'.txt', 'r') 
                Lines = file1.readlines()

                for line in Lines:
                    a = line.split(',')
                    if a[0] == x:
                        x1 = int(float(a[1]))
                        y1 = int(float(a[2]))
                        x2 = int(float(a[3]))
                        y2 = int(float(a[4]))
                        flag = 1
                        break
                
                if flag == 1:
                    break
              
            Rx = 256/178
            Ry = 256/218
            
            bbox = (int(Ry*y1), int(Rx*x1), int(Ry*(y2-y1)), int(Rx*(x2-x1)))#util.bbox(self.opt)
            
            print(bbox)
            mask = util.bbox2mask(bbox, self.opt)  # Tensor, [1, H, W]
            bbox_t = torch.from_numpy(np.array(bbox))
            return bbox_t, mask

        elif self.opt['mask_type'] == 'irregular':  # irregular mask
            if self.opt['use_shuffle']:  # train or val (not include test)
                mask_index = random.randint(0, len(self.mask_data) - 1)
            else:
                mask_index = index
            mask = self.loader(self.mask_data[mask_index]).convert('L')  # image [H, W, 1], [0, 255]
            mask = np.asarray(mask.resize((self.opt['img_shape'][2], self.opt['img_shape'][1]), Image.BICUBIC))
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(np.expand_dims(mask, 0))  # Tensor, [1, H, W]
            return mask
        else:
            raise NotImplementedError('Unsupported mask type: {}'.format(self.opt['mask_type']))

    def resize(self, img, height, width, centerCrop=False):  # mainly for celeba dataset | place365 | paris_streetview
        imgw, imgh = img.size[0], img.size[1]  # [w, h, c]

        '''if imgh != imgw:

            if centerCrop:
                # center crop, mainly for celeba
                side = np.minimum(imgh, imgw)
                j = (imgh - side) // 2
                i = (imgw - side) // 2
                img = img.crop((i, j, side, side))
            else:
                # random crop, mainly for place365 and paris_streetview
                side = np.minimum(imgh, imgw)
                ix = random.randrange(0, imgw - side + 1)
                iy = random.randrange(0, imgh - side + 1)
                img = img.crop((ix, iy, side, side))
        '''
        Rx = imgw/width
        Ry = imgh/height
        img = img.resize((width, height), Image.BICUBIC)

        return img
