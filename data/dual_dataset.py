import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class DualDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)
        self.ABC_paths = sorted(make_dataset(self.dir_ABC))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        ABC_path = self.ABC_paths[index]
        ABC = Image.open(ABC_path).convert('RGB')
        ABC = ABC.resize((self.opt.loadSize * 3, self.opt.loadSize), Image.BICUBIC)
        ABC = transforms.ToTensor()(ABC)

        w_total = ABC.size(2)
        w = int(w_total / 3)
        h = ABC.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = ABC[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = ABC[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]
        C = ABC[:, h_offset:h_offset + self.opt.fineSize,
               w + w + w_offset:w + w + w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)

        BC = torch.cat((B, C), 0)


        # opt.target = where to collapse
        # collapse B and C


        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': BC,
                'A_paths': ABC_path, 'B_paths': ABC_path}

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'DualDataset'
