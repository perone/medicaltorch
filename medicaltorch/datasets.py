import os
import collections

from medicaltorch import transforms as mt_transforms

from tqdm import tqdm
import numpy as np
import nibabel as nib

from torch.utils.data import Dataset
import torch
from torch._six import string_classes, int_classes

from PIL import Image


__numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class SampleMetadata(object):
    def __init__(self, d=None):
        self.metadata = {} or d

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def __getitem__(self, key):
        return self.metadata[key]

    def __contains__(self, key):
        return key in self.metadata

    def keys(self):
        return self.metadata.keys()


class BatchSplit(object):
    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        batch_len = len(self.batch["input"])
        for i in range(batch_len):
            single_sample = {k: v[i] for k, v in self.batch.items()}
            single_sample['index'] = i
            yield single_sample
        raise StopIteration


class SegmentationPair2D(object):
    """This class is used to build 2D segmentation datasets. It represents
    a pair of of two data volumes (the input data and the ground truth data).

    :param input_filename: the input filename (supported by nibabel).
    :param gt_filename: the ground-truth filename.
    :param cache: if the data should be cached in memory or not.
    :param canonical: canonical reordering of the volume axes.
    """
    def __init__(self, input_filename, gt_filename, cache=True,
                 canonical=False):
        self.input_filename = input_filename
        self.gt_filename = gt_filename
        self.canonical = canonical
        self.cache = cache

        self.input_handle = nib.load(self.input_filename)

        # Unlabeled data (inference time)
        if self.gt_filename is None:
            self.gt_handle = None
        else:
            self.gt_handle = nib.load(self.gt_filename)

        if len(self.input_handle.shape) > 3:
            raise RuntimeError("4-dimensional volumes not supported.")

        # Sanity check for dimensions, should be the same
        input_shape, gt_shape = self.get_pair_shapes()

        if self.gt_handle is not None:
            if not np.allclose(input_shape, gt_shape):
                raise RuntimeError('Input and ground truth with different dimensions.')

        if self.canonical:
            self.input_handle = nib.as_closest_canonical(self.input_handle)

            # Unlabeled data
            if self.gt_handle is not None:
                self.gt_handle = nib.as_closest_canonical(self.gt_handle)

    def get_pair_shapes(self):
        """Return the tuple (input, ground truth) representing both the input
        and ground truth shapes."""
        input_shape = self.input_handle.header.get_data_shape()

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_shape = None
        else:
            gt_shape = self.gt_handle.header.get_data_shape()

        return input_shape, gt_shape

    def get_pair_data(self):
        """Return the tuble (input, ground truth) with the data content in
        numpy array."""
        cache_mode = 'fill' if self.cache else 'unchanged'
        input_data = self.input_handle.get_fdata(cache_mode, dtype=np.float32)

        # Handle unlabeled data
        if self.gt_handle is None:
            gt_data = None
        else:
            gt_data = self.gt_handle.get_fdata(cache_mode, dtype=np.float32)

        return input_data, gt_data

    def get_pair_slice(self, slice_index, slice_axis=2):
        """Return the specified slice from (input, ground truth).

        :param slice_index: the slice number.
        :param slice_axis: axis to make the slicing.
        """
        if self.cache:
            input_dataobj, gt_dataobj = self.get_pair_data()
        else:
            # use dataobj to avoid caching
            input_dataobj = self.input_handle.dataobj

            if self.gt_handle is None:
                gt_dataobj = None
            else:
                gt_dataobj = self.gt_handle.dataobj

        if slice_axis not in [0, 1, 2]:
            raise RuntimeError("Invalid axis, must be between 0 and 2.")

        if slice_axis == 2:
            input_slice = np.asarray(input_dataobj[..., slice_index],
                                     dtype=np.float32)
        elif slice_axis == 1:
            input_slice = np.asarray(input_dataobj[:, slice_index, ...],
                                     dtype=np.float32)
        elif slice_axis == 0:
            input_slice = np.asarray(input_dataobj[slice_index, ...],
                                     dtype=np.float32)

        # Handle the case for unlabeled data
        gt_meta_dict = None
        if self.gt_handle is None:
            gt_slice = None
        else:
            if slice_axis == 2:
                gt_slice = np.asarray(gt_dataobj[..., slice_index],
                                      dtype=np.float32)
            elif slice_axis == 1:
                gt_slice = np.asarray(gt_dataobj[:, slice_index, ...],
                                      dtype=np.float32)
            elif slice_axis == 0:
                gt_slice = np.asarray(gt_dataobj[slice_index, ...],
                                      dtype=np.float32)

            gt_meta_dict = SampleMetadata({
                "zooms": self.gt_handle.header.get_zooms()[:2],
                "data_shape": self.gt_handle.header.get_data_shape()[:2],
            })

        input_meta_dict = SampleMetadata({
            "zooms": self.input_handle.header.get_zooms()[:2],
            "data_shape": self.input_handle.header.get_data_shape()[:2],
        })

        dreturn = {
            "input": input_slice,
            "gt": gt_slice,
            "input_metadata": input_meta_dict,
            "gt_metadata": gt_meta_dict,
        }
        
        return dreturn


class MRI2DSegmentationDataset(Dataset):
    """This is a generic class for 2D (slice-wise) segmentation datasets.

    :param filename_pairs: a list of tuples in the format (input filename,
                           ground truth filename).
    :param slice_axis: axis to make the slicing (default axial).
    :param cache: if the data should be cached in memory or not.
    :param transform: transformations to apply.
    """
    def __init__(self, filename_pairs, slice_axis=2, cache=True,
                 transform=None, slice_filter_fn=None, canonical=False):
        self.filename_pairs = filename_pairs
        self.handlers = []
        self.indexes = []
        self.transform = transform
        self.cache = cache
        self.slice_axis = slice_axis
        self.slice_filter_fn = slice_filter_fn
        self.canonical = canonical

        self._load_filenames()
        self._prepare_indexes()

    def _load_filenames(self):
        for input_filename, gt_filename in self.filename_pairs:
            segpair = SegmentationPair2D(input_filename, gt_filename,
                                         self.cache, self.canonical)
            self.handlers.append(segpair)
        
    def _prepare_indexes(self):
        for segpair in self.handlers:
            input_data_shape, _ = segpair.get_pair_shapes()
            for segpair_slice in range(input_data_shape[2]):

                # Check if slice pair should be used or not
                if self.slice_filter_fn:
                    slice_pair = segpair.get_pair_slice(segpair_slice,
                                                        self.slice_axis)
                    
                    filter_fn_ret = self.slice_filter_fn(slice_pair)
                    if not filter_fn_ret:
                        continue

                item = (segpair, segpair_slice)
                self.indexes.append(item)

    def set_transform(self, transform):
        """This method will replace the current transformation for the
        dataset.

        :param transform: the new transformation
        """
        self.transform = transform

    def compute_mean_std(self, verbose=False):
        """Compute the mean and standard deviation of the entire dataset.
        
        :param verbose: if True, it will show a progress bar.
        :returns: tuple (mean, std dev) 
        """
        sum_intensities = 0.0
        numel = 0

        with DatasetManager(self,
                            override_transform=mt_transforms.ToTensor()) as dset:
            pbar = tqdm(dset, desc="Mean calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_intensities += input_data.sum()
                numel += input_data.numel()
                pbar.set_postfix(mean="{:.2f}".format(sum_intensities / numel),
                                 refresh=False)

            training_mean = sum_intensities / numel

            sum_var = 0.0
            numel = 0

            pbar = tqdm(dset, desc="Std Dev calculation", disable=not verbose)
            for sample in pbar:
                input_data = sample['input']
                sum_var += (input_data - training_mean).pow(2).sum()
                numel += input_data.numel()
                pbar.set_postfix(std="{:.2f}".format(np.sqrt(sum_var / numel)),
                                 refresh=False)

        training_std = np.sqrt(sum_var / numel)
        return training_mean.item(), training_std.item()

    def __len__(self):
        """Return the dataset size."""
        return len(self.indexes)

    def __getitem__(self, index):
        """Return the specific index pair slices (input, ground truth).

        :param index: slice index.
        """
        segpair, segpair_slice = self.indexes[index]
        pair_slice = segpair.get_pair_slice(segpair_slice,
                                            self.slice_axis)

        # Consistency with torchvision, returning PIL Image
        # Using the "Float mode" of PIL, the only mode
        # supporting unbounded float32 values
        input_img = Image.fromarray(pair_slice["input"], mode='F')

        # Handle unlabeled data
        if pair_slice["gt"] is None:
            gt_img = None
        else:
            gt_img = Image.fromarray(pair_slice["gt"], mode='F')

        data_dict = {
            'input': input_img,
            'gt': gt_img,
            'input_metadata': pair_slice['input_metadata'],
            'gt_metadata': pair_slice['gt_metadata'],
        }

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict


class DatasetManager(object):
    def __init__(self, dataset, override_transform=None):
        self.dataset = dataset
        self.override_transform = override_transform
        self._transform_state = None

    def __enter__(self):
        if self.override_transform:
            self._transform_state = self.dataset.transform
            self.dataset.transform = self.override_transform
        return self.dataset

    def __exit__(self, *args):
        if self._transform_state:
            self.dataset.transform = self._transform_state


class SCGMChallenge2DTrain(MRI2DSegmentationDataset):
    """This is the Spinal Cord Gray Matter Challenge dataset.

    :param root_dir: the directory containing the training dataset.
    :param site_ids: a list of site ids to filter (i.e. [1, 3]).
    :param subj_ids: the list of subject ids to filter.
    :param rater_ids: the list of the rater ids to filter.
    :param transform: the transformations that should be applied.
    :param cache: if the data should be cached in memory or not.
    :param slice_axis: axis to make the slicing (default axial).

    .. note:: This dataset assumes that you only have one class in your
              ground truth mask (w/ 0's and 1's). It also doesn't
              automatically resample the dataset.

    .. seealso::
        Prados, F., et al (2017). Spinal cord grey matter
        segmentation challenge. NeuroImage, 152, 312–329.
        https://doi.org/10.1016/j.neuroimage.2017.03.010

        Challenge Website:
        http://cmictig.cs.ucl.ac.uk/spinal-cord-grey-matter-segmentation-challenge
    """
    NUM_SITES = 4
    NUM_SUBJECTS = 10
    NUM_RATERS = 4

    def __init__(self, root_dir, slice_axis=2, site_ids=None,
                 subj_ids=None, rater_ids=None, cache=True,
                 transform=None, slice_filter_fn=None,
                 canonical=False, labeled=True):

        self.labeled = labeled
        self.root_dir = root_dir
        self.site_ids = site_ids or range(1, SCGMChallenge2DTrain.NUM_SITES + 1)
        self.subj_ids = subj_ids or range(1, SCGMChallenge2DTrain.NUM_SUBJECTS + 1)
        self.rater_ids = rater_ids or range(1, SCGMChallenge2DTrain.NUM_RATERS + 1)

        self.filename_pairs = []

        for site_id in self.site_ids:
            for subj_id in self.subj_ids:
                if len(self.rater_ids) > 0:
                    for rater_id in self.rater_ids:
                        input_filename = self._build_train_input_filename(site_id, subj_id)
                        gt_filename = self._build_train_input_filename(site_id, subj_id, rater_id)

                        input_filename = os.path.join(self.root_dir, input_filename)
                        gt_filename = os.path.join(self.root_dir, gt_filename)

                        if not self.labeled:
                            gt_filename = None

                        self.filename_pairs.append((input_filename, gt_filename))
                else:
                    input_filename = self._build_train_input_filename(site_id, subj_id)
                    gt_filename = None
                    input_filename = os.path.join(self.root_dir, input_filename)

                    if not self.labeled:
                        gt_filename = None
                        
                    self.filename_pairs.append((input_filename, gt_filename))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)

    @staticmethod
    def _build_train_input_filename(site_id, subj_id, rater_id=None):
        if rater_id is None:
            return "site{:d}-sc{:02d}-image.nii.gz".format(site_id, subj_id)
        else:
            return "site{:d}-sc{:02d}-mask-r{:d}.nii.gz".format(site_id, subj_id, rater_id)


class SCGMChallenge2DTest(MRI2DSegmentationDataset):
    """This is the Spinal Cord Gray Matter Challenge dataset.

    :param root_dir: the directory containing the test dataset.
    :param site_ids: a list of site ids to filter (i.e. [1, 3]).
    :param subj_ids: the list of subject ids to filter.
    :param transform: the transformations that should be applied.
    :param cache: if the data should be cached in memory or not.
    :param slice_axis: axis to make the slicing (default axial).

    .. note:: This dataset assumes that you only have one class in your
              ground truth mask (w/ 0's and 1's). It also doesn't
              automatically resample the dataset.

    .. seealso::
        Prados, F., et al (2017). Spinal cord grey matter
        segmentation challenge. NeuroImage, 152, 312–329.
        https://doi.org/10.1016/j.neuroimage.2017.03.010

        Challenge Website:
        http://cmictig.cs.ucl.ac.uk/spinal-cord-grey-matter-segmentation-challenge
    """
    NUM_SITES = 4
    NUM_SUBJECTS = 10

    def __init__(self, root_dir, slice_axis=2, site_ids=None,
                 subj_ids=None, cache=True,
                 transform=None, slice_filter_fn=None,
                 canonical=False):

        self.root_dir = root_dir
        self.site_ids = site_ids or range(1, SCGMChallenge2DTest.NUM_SITES + 1)
        self.subj_ids = subj_ids or range(11, 10 + SCGMChallenge2DTest.NUM_SUBJECTS + 1)

        self.filename_pairs = []

        for site_id in self.site_ids:
            for subj_id in self.subj_ids:
                input_filename = self._build_train_input_filename(site_id, subj_id)
                gt_filename = None

                input_filename = os.path.join(self.root_dir, input_filename)
                if not os.path.exists(input_filename):
                    raise RuntimeError("Path '{}' doesn't exist !".format(input_filename))
                self.filename_pairs.append((input_filename, gt_filename))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)

    @staticmethod
    def _build_train_input_filename(site_id, subj_id, rater_id=None):
        if rater_id is None:
            return "site{:d}-sc{:02d}-image.nii.gz".format(site_id, subj_id)
        else:
            return "site{:d}-sc{:02d}-mask-r{:d}.nii.gz".format(site_id, subj_id, rater_id)



def mt_collate(batch):
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        stacked = torch.stack(batch, 0)
        return stacked
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return __numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: mt_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [mt_collate(samples) for samples in transposed]

    return batch

