import os
import pytest

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medicaltorch import datasets as mt_datasets
from medicaltorch import transforms as mt_transforms

ROOT_DIR_GMCHALLENGE = './data'


class TestMRIDataset(object):
    @pytest.fixture
    def gmsample_files(self):
        mri_input_filename = os.path.join(ROOT_DIR_GMCHALLENGE,
                                          'site1-sc01-image.nii.gz')
        mri_gt_filename = os.path.join(ROOT_DIR_GMCHALLENGE,
                                       'site1-sc01-mask-r1.nii.gz')
        return mri_input_filename, mri_gt_filename

    def test_pair_loading(self, gmsample_files):
        mri_input_filename, mri_gt_filename = gmsample_files
        pair = mt_datasets.SegmentationPair2D(mri_input_filename,
                                              mri_gt_filename)
        assert pair is not None

    def test_pair_slicing(self, gmsample_files):
        mri_input_filename, mri_gt_filename = gmsample_files
        pair = mt_datasets.SegmentationPair2D(mri_input_filename,
                                              mri_gt_filename)
        slice_pair = pair.get_pair_slice(0)
        input_slice = slice_pair["input"]
        gt_slice = slice_pair["gt"]
        
        assert input_slice.shape == (200, 200)
        assert gt_slice.shape == input_slice.shape
        assert input_slice[0][0] == pytest.approx(651.5, 0.1)

    def test_dataset(self, gmsample_files):
        mri_input_filename, mri_gt_filename = gmsample_files
        filename_mapping = [(mri_input_filename, mri_gt_filename)]
        dataset = mt_datasets.MRI2DSegmentationDataset(filename_mapping)
        assert len(dataset) == 3

        first_item = dataset[0]
        assert isinstance(first_item, dict)
        assert 'input' in first_item
        assert 'gt' in first_item

        input_slice = first_item['input']
        assert input_slice.getpixel((0, 0)) == pytest.approx(651.5, 0.1)

    def test_dataset_transform(self, gmsample_files):
        mri_input_filename, mri_gt_filename = gmsample_files
        filename_mapping = [(mri_input_filename, mri_gt_filename)]
        dataset = mt_datasets.MRI2DSegmentationDataset(filename_mapping,
                                                       transform=mt_transforms.ToTensor())
        first_item = dataset[0]
        assert isinstance(first_item['input'], torch.FloatTensor)
        assert first_item['input'].size() == (1, 200, 200)
        assert first_item['input'].size() == first_item['gt'].size()

    def test_dataset_loader(self, gmsample_files):
        mri_input_filename, mri_gt_filename = gmsample_files
        filename_mapping = [(mri_input_filename, mri_gt_filename)]
        dataset = mt_datasets.MRI2DSegmentationDataset(filename_mapping,
                                                       transform=mt_transforms.ToTensor())

        dataloader = DataLoader(dataset, batch_size=4,
                                shuffle=True, num_workers=1,
                                collate_fn=mt_datasets.mt_collate)

        minibatch = next(iter(dataloader))
        assert len(minibatch) == 4
        assert 'input' in minibatch
        assert 'gt' in minibatch
        assert 'input_metadata' in minibatch
        assert 'gt_metadata' in minibatch

    def test_gmchallenge_dataset(self):
        composed_transform = transforms.Compose([
            mt_transforms.CenterCrop2D((200, 200)),
            mt_transforms.ToTensor(),
        ])

        dataset = mt_datasets.SCGMChallenge2D(root_dir=ROOT_DIR_GMCHALLENGE,
                                              transform=composed_transform)
        assert len(dataset) == 2204

        dataset = mt_datasets.SCGMChallenge2D(root_dir=ROOT_DIR_GMCHALLENGE,
                                              rater_ids=[4, ], subj_ids=[1, 2],
                                              transform=composed_transform)
        assert len(dataset) == 107

        dataloader = DataLoader(dataset, batch_size=4,
                                shuffle=True, num_workers=4,
                                collate_fn=mt_datasets.mt_collate)
        minibatch = next(iter(dataloader))
        assert len(minibatch) == 4
        assert minibatch['input'].size() == (4, 1, 200, 200)

        iterations = 0
        for minbatch in dataloader:
            iterations += 1
        assert iterations == 27
