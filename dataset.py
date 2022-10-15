import torch
import numpy as np
import torch.utils.data as data
import h5py
import os


class VMVPH5(data.Dataset):
    def __init__(self, train=True, npoints=2048):
        if train:
            self.input_path = './data/vmvp_train_input.h5'
            self.gt_path = './data/vmvp_train_gt_%dpts.h5' % npoints
        else:
            self.input_path = './data/vmvp_test_input.h5'
            self.gt_path = './data/vmvp_test_gt_%dpts.h5' % npoints
        self.npoints = npoints
        self.train = train

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()])).astype(np.float32)
        self.labels = np.array((input_file['labels'][()]))
        self.view_points = np.array((input_file['view_points'][()])).astype(np.float32)
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()])).astype(np.float32)
        gt_file.close()

        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // 26]))
        view_point = torch.from_numpy((self.view_points[index]))
        label = (self.labels[index])
        return label, view_point, partial, complete


class VP_VMVPH5(data.Dataset):
    def __init__(self, train=True):
        if train:
            self.input_path = './data/vmvp_train_input.h5'
        else:
            self.input_path = './data/vmvp_test_input.h5'
        self.train = train

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()])).astype(np.float32)
        self.labels = np.array((input_file['labels'][()]))
        self.view_points = np.array((input_file['view_points'][()])).astype(np.float32)
        input_file.close()

        print(self.input_data.shape)
        print(self.labels.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        view_point = torch.from_numpy((self.view_points[index]))
        label = (self.labels[index])
        return label, view_point, partial

