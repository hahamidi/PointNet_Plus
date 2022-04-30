import os
import shlex
import subprocess

import h5py
import numpy as np
from sympy import I
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    
    f = h5py.File(name, "r")
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class Indoor3DSemSeg(data.Dataset):
    def __init__(self, num_points, train=True,test_area = [5,6], download=False, data_precent=1.0):
        super().__init__()
        self.data_precent = data_precent
        self.folder = "indoor3d_sem_seg_hdf5_data"
        self.data_dir = os.path.join(BASE_DIR, self.folder)
        self.url = (
            "https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip"
        )


        self.train, self.num_points = train, num_points

        all_files = _get_data_files(os.path.join(self.data_dir, "all_files.txt"))
        room_filelist = _get_data_files(
            os.path.join(self.data_dir, "room_filelist.txt")
        )

        data_batchlist, label_batchlist = [], []
        for f in all_files:
            data, label = _load_data_file(os.path.join(BASE_DIR, f))
            data_batchlist.append(data)
            label_batchlist.append(label)

        data_batches = np.concatenate(data_batchlist, 0)
        labels_batches = np.concatenate(label_batchlist, 0)
        test_area_name = ["Area_"+str(num) for num in test_area]
        train_idxs, test_idxs = [], []
        for i, room_name in enumerate(room_filelist):
            area_name =  "_".join(room_name.split("_")[0:2])
            if area_name in test_area_name:
                test_idxs.append(i)            
            else:
                train_idxs.append(i)

        if self.train:
            self.points = data_batches[train_idxs, ...]
            self.labels = labels_batches[train_idxs, ...]
        else:
            self.points = data_batches[test_idxs, ...]
            self.labels = labels_batches[test_idxs, ...]

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points = torch.from_numpy(self.points[idx, pt_idxs].copy()).float()
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()).long()

        return current_points, current_labels

    def __len__(self):
        return int(self.points.shape[0] * self.data_precent)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass
class fakeIndoor3DSemSeg(data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.size = 128
    def __getitem__(self,idx):
        current_points = torch.rand(4096,9).float()
        current_labels = torch.randint(0, 13, (4096,)).long()
        return current_points, current_labels
    def __len__(self):
        return self.size
    

if __name__ == "__main__":
    # dset = Indoor3DSemSeg(128, "./")
    # print(dset[0])
    # print(len(dset))
    # dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    # for i, data in enumerate(dloader, 0):
    #     inputs, labels = data
    #     if i == len(dloader) - 1:
    #         print(inputs.size())
    
    dset = fakeIndoor3DSemSeg()
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())