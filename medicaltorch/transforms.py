import skimage
import numpy as np
import numbers
import torchvision.transforms.functional as F
from scipy.ndimage import center_of_mass
from torchvision import transforms
import torch
from PIL import Image

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


class MTTransform(object):

    def __call__(self, sample):
        raise NotImplementedError("You need to implement the transform() method.")

    def undo_transform(self, sample):
        raise NotImplementedError("You need to implement the undo_transform() method.")


class UndoCompose(object):
    def __init__(self, compose):
        self.transforms = compose.transforms

    def __call__(self):
        for t in self.transforms:
            img = t.undo_transform(img)
        return img


class UndoTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        return self.transform.undo_transform(sample)


class ToTensor(MTTransform):
    """Convert a PIL image(s) or numpy array(s) to a PyTorch tensor(s)."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        if len(input_data) > 1:
            # Multiple inputs
            ret_input = [F.to_tensor(item) for item in input_data]
        else:
            # single input
            ret_input = F.to_tensor(input_data[0])

            # transform list of dic into single dic
            rdict['input_metadata'] = sample['input_metadata'][0]

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                if isinstance(gt_data, list):
                    # multiple GT
                    ret_gt = [F.to_tensor(item) for item in gt_data]
                else:
                    # single GT
                    ret_gt = F.to_tensor(gt_data)

                rdict['gt'] = ret_gt
        sample.update(rdict)
        return sample


class ToPIL(MTTransform):
    def __init__(self, labeled=True):
        self.labeled = labeled

    def sample_transform(self, sample_data):
        # Numpy array
        if not isinstance(sample_data, np.ndarray):
            input_data_npy = sample_data.numpy()
        else:
            input_data_npy = sample_data

        input_data_npy = np.transpose(input_data_npy, (1, 2, 0))
        input_data_npy = np.squeeze(input_data_npy, axis=2)
        input_data = Image.fromarray(input_data_npy, mode='F')
        return input_data

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        if isinstance(input_data, list):
            ret_input = [self.sample_transform(item) for item in input_data]
        else:
            ret_input = self.sample_transform(input_data)

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']

            if isinstance(gt_data, list):
                ret_gt = [self.sample_transform(item) for item in gt_data]
            else:
                ret_gt = self.sample_transform(gt_data)

            rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


class StackTensors(MTTransform):
    """
    Stack all modalities in a single vector.

    TODO: add reverse transformation
    """

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        rdict['input'] = torch.squeeze(torch.cat(input_data, dim=0))
        sample.update(rdict)
        return sample


class Crop2D(MTTransform):
    """Make a center crop of a specified size.

    :param segmentation: if it is a segmentation task.
                         When this is True (default), the crop
                         will also be applied to the ground truth.
    """

    def __init__(self, size, labeled=True):
        self.size = size
        self.labeled = labeled

    @staticmethod
    def propagate_params(sample, params, i):
        input_metadata = sample['input_metadata'][i]
        input_metadata["__centercrop"] = params
        return input_metadata

    @staticmethod
    def get_params(sample):
        return [sample['input_metadata'][i]["__centercrop"] for i in range(len(sample))]

    def undo_transform(self, sample):
        rdict = {}
        input_data = sample['input']
        params = self.get_params(sample)
        th, tw = self.size
        for i in range(len(input_data)):

            fh, fw, w, h = params[i]

            pad_left = fw
            pad_right = w - pad_left - tw
            pad_top = fh
            pad_bottom = h - pad_top - th

            padding = (pad_left, pad_top, pad_right, pad_bottom)
            input_data[i] = F.pad(input_data[i], padding)

        rdict['input'] = input_data

        sample.update(rdict)
        return sample


class CenterCrop2D(Crop2D):
    """Make a centered crop of a specified size.
    :param labeled: if it is a segmentation task.
                         When this is True (default), the crop
                         will also be applied to the ground truth.
    """

    def __init__(self, size, labeled=True):
        super().__init__(size, labeled)

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        th, tw = self.size

        for i in range(len(input_data)):
            w, h = input_data[i].size

            fh = int(round((h - th) / 2.))
            fw = int(round((w - tw) / 2.))
            params = (fh, fw, w, h)

            # Updating the parameters in the input metadata
            self.propagate_params(sample, params, i)
            # Cropping
            input_data[i] = F.center_crop(input_data[i], self.size)

        rdict['input'] = input_data

        if self.labeled:
            gt_data = sample['gt']
            gt_metadata = sample['gt_metadata']

            w, h = gt_data.size
            fh = int(round((h - th) / 2.))
            fw = int(round((w - tw) / 2.))

            gt_data = F.center_crop(gt_data, self.size)
            gt_metadata["__centercrop"] = (fh, fw, w, h)
            rdict['gt'] = gt_data

        sample.update(rdict)
        return sample

    # Reverse transformation. Implemented by @Charleygros
    def _uncrop(self, data, params):
        fh, fw, w, h = params
        th, tw = self.size
        pad_left = fw
        pad_right = w - pad_left - tw
        pad_top = fh
        pad_bottom = h - pad_top - th
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        return F.pad(data, padding)

    def undo_transform(self, sample):
        rdict = {}
        if isinstance(sample['input'], list):
            for i in range(len(sample['input'])):
                rdict['input'] = self._uncrop(sample['input'][i], sample['input_metadata'][i]["__centercrop"])
        else:
            rdict['input'] = self._uncrop(sample['input'], sample['input_metadata']["__centercrop"])

        rdict['gt'] = self._uncrop(sample['gt'], sample['gt_metadata']["__centercrop"])
        sample.update(rdict)
        return sample


class ROICrop2D(Crop2D):
    """Make a crop of a specified size around a ROI.
    :param labeled: if it is a segmentation task.
                         When this is True (default), the crop
                         will also be applied to the ground truth.
    """

    def __init__(self, size, labeled=True):
        super().__init__(size, labeled)

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        roi_data = sample['roi']

        # compute center of mass of the ROI
        x_roi, y_roi = center_of_mass(np.array(roi_data).astype(np.int))
        x_roi, y_roi = int(round(x_roi)), int(round(y_roi))

        for i in range(len(input_data)):
            w, h = input_data[i].size
            th, tw = self.size
            th_half, tw_half = int(round(th / 2.)), int(round(tw / 2.))

            # compute top left corner of the crop area
            fh = y_roi - th_half
            fw = x_roi - tw_half
            params = (fh, fw, w, h)

            self.propagate_params(sample, params, i)

            # crop data
            input_data[i] = F.crop(input_data[i], fw, fh, tw, th)
        rdict['input'] = input_data

        if self.labeled:
            gt_data = sample['gt']
            gt_metadata = sample['gt_metadata']
            gt_data = F.crop(gt_data, fw, fh, tw, th)
            gt_metadata["__centercrop"] = (fh, fw, w, h)
            rdict['gt'] = gt_data

        sample.update(rdict)
        return sample


class Normalize(MTTransform):
    """Normalize a tensor image with mean and standard deviation.

    :param mean: mean value.
    :param std: standard deviation value.

    In case of multiple inputs, both mean and std are lists.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        input_data = sample['input']
        if isinstance(input_data, list):
            input_data = [F.normalize(input_data[i], self.mean[i], self.std[i]) for i in range(len(input_data))]
        else:
            input_data = F.normalize(input_data, self.mean, self.std)

        rdict = {
            'input': input_data,
        }
        sample.update(rdict)
        return sample


class NormalizeInstance(MTTransform):
    """Normalize a tensor image with mean and standard deviation estimated
    from the sample itself.

    :param mean: mean value.
    :param std: standard deviation value.
    """

    def __call__(self, sample):
        input_data = sample['input']
        if isinstance(input_data, list):
            for i in range(len(input_data)):
                mean, std = input_data[i].mean(), input_data[i].std()
                input_data[i] = F.normalize(input_data[i], [mean], [std])
        else:
            mean, std = input_data.mean(), input_data.std()
            input_data = F.normalize(input_data, [mean], [std])
        
        rdict = {
            'input': input_data,
        }
        sample.update(rdict)
        return sample


class NormalizeInstance3D(MTTransform):
    """Normalize a tensor volume with mean and standard deviation estimated
    from the sample itself.

    :param mean: mean value.
    :param std: standard deviation value.
    """

    def __call__(self, sample):
        input_data_normalized = []
        input_data = sample['input']
        if isinstance(input_data, list):
            for i in range(len(input_data)):
                input_volume = input_data[i]
                mean, std = input_volume.mean(), input_volume.std()
                if mean != 0 or std != 0:
                    input_data_normalized.append(F.normalize(input_volume,
                                                             [mean for _ in range(0, input_volume.shape[0])],
                                                             [std for _ in range(0, input_volume.shape[0])]).unsqueeze(0))

        else:
            mean, std = input_data.mean(), input_data.std()

            if mean != 0 or std != 0:
                input_volume = input_data
                input_data_normalized = F.normalize(input_volume,
                                                    [mean for _ in range(0, input_volume.shape[0])],
                                                    [std for _ in range(0, input_volume.shape[0])]).unsqueeze(0)
        rdict = {
            'input': input_data_normalized,
            'gt': sample['gt'].unsqueeze(0)
        }
        sample.update(rdict)
        return sample


class RandomRotation(MTTransform):
    def __init__(self, degrees, resample=False,
                 expand=False, center=None,
                 labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.labeled = labeled

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input'][0]
        angle = self.get_params(self.degrees)
        for i in range(len(input_data)):
            input_data[i] = F.rotate(input_data[i], angle,
                                     self.resample, self.expand,
                                     self.center)
        rdict['input'] = input_data

        if self.labeled:
            gt_data = sample['gt']
            gt_data = F.rotate(gt_data, angle,
                               self.resample, self.expand,
                               self.center)
            rdict['gt'] = gt_data

        sample.update(rdict)
        return sample


class RandomRotation3D(MTTransform):
    """Make a rotation of the volume's values.

    :param degrees: Maximum rotation's degrees.
    :param axis: Axis of the rotation.
    """

    def __init__(self, degrees, axis=0, labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.labeled = labeled
        self.axis = axis

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        if len(sample['input'][0].shape) != 3:
            raise ValueError("Input of RandomRotation3D should be a 3 dimensionnal tensor.")

        angle = self.get_params(self.degrees)
        input_rotated = [np.zeros(input_data[0].shape, dtype=input_data.dtype) for i in range(len(input_data))]
        gt_data = sample['gt'] if self.labeled else None
        gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None

        # TODO: Would be faster with only one vectorial operation
        # TODO: Use the axis index for factoring this loopgit a
        for x in range(input_data.shape[self.axis]):
            if self.axis == 0:
                input_rotated[x, :, :] = F.rotate(Image.fromarray(input_data[x, :, :], mode='F'), angle)
                if self.labeled:
                    gt_rotated[x, :, :] = F.rotate(Image.fromarray(gt_data[x, :, :], mode='F'), angle)
            if self.axis == 1:
                input_rotated[:, x, :] = F.rotate(Image.fromarray(input_data[:, x, :], mode='F'), angle)
                if self.labeled:
                    gt_rotated[:, x, :] = F.rotate(Image.fromarray(gt_data[:, x, :], mode='F'), angle)
            if self.axis == 2:
                input_rotated[:, :, x] = F.rotate(Image.fromarray(input_data[:, :, x], mode='F'), angle)
                if self.labeled:
                    gt_rotated[:, :, x] = F.rotate(Image.fromarray(gt_data[:, :, x], mode='F'), angle)

        rdict['input'] = input_rotated
        if self.labeled:
            rdict['gt'] = gt_rotated
        sample.update(rdict)

        return sample


class RandomReverse3D(MTTransform):
    """Make a symmetric inversion of the different values of each dimensions.
    (randomized)
    """

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_list = sample['input']
        if not isinstance(input_list, list):
            input_list = [sample['input']]
        gt_data = sample['gt'] if self.labeled else None
        reverse_input = []
        for input_data in input_list:
            if np.random.randint(2) == 1:
                input_data = np.flip(input_data, axis=0).copy()
                if self.labeled: gt_data = np.flip(gt_data, axis=0).copy()
            if np.random.randint(2) == 1:
                input_data = np.flip(input_data, axis=1).copy()
                if self.labeled: gt_data = np.flip(gt_data, axis=1).copy()
            if np.random.randint(2) == 1:
                input_data = np.flip(input_data, axis=2).copy()
                if self.labeled: gt_data = np.flip(gt_data, axis=2).copy()
            reverse_input.append(input_data)

        rdict['input'] = reverse_input
        if self.labeled: rdict['gt'] = gt_data

        sample.update(rdict)
        return sample


class RandomAffine(MTTransform):
    def __init__(self, degrees, translate=None,
                 scale=None, shear=None,
                 resample=False, fillcolor=0,
                 labeled=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.labeled = labeled

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation
        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = np.random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = np.random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def sample_augment(self, input_data, params):
        input_data = F.affine(input_data, *params, resample=self.resample,
                              fillcolor=self.fillcolor)
        return input_data

    def label_augment(self, gt_data, params):
        gt_data = self.sample_augment(gt_data, params)
        np_gt_data = np.array(gt_data)
        np_gt_data[np_gt_data >= 0.5] = 255.0
        np_gt_data[np_gt_data < 0.5] = 0.0
        np_gt_data = np_gt_data.astype(np.uint8)
        gt_data = Image.fromarray(np_gt_data, mode='L')
        return gt_data

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        rdict = {}
        input_data = sample['input']

        input_data_size = input_data[0].size

        params = self.get_params(self.degrees, self.translate, self.scale,
                                 self.shear, input_data_size)

        if isinstance(input_data, list):
            ret_input = [self.sample_augment(item, params)
                         for item in input_data]
        else:
            ret_input = self.sample_augment(input_data, params)

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if isinstance(gt_data, list):
                ret_gt = [self.label_augment(item, params)
                          for item in gt_data]
            else:
                ret_gt = self.label_augment(gt_data, params)

            rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


class RandomTensorChannelShift(MTTransform):
    def __init__(self, shift_range):
        self.shift_range = shift_range

    @staticmethod
    def get_params(shift_range):
        sampled_value = np.random.uniform(shift_range[0],
                                          shift_range[1])
        return sampled_value

    def sample_augment(self, input_data, params):
        np_input_data = np.array(input_data)
        np_input_data += params
        input_data = Image.fromarray(np_input_data, mode='F')
        return input_data

    def __call__(self, sample):
        input_data = sample['input']
        params = self.get_params(self.shift_range)

        if isinstance(input_data, list):
            ret_input = [self.sample_augment(item, params) for item in input_data]
        else:
            ret_input = self.sample_augment(input_data, params)

        rdict = {
            'input': ret_input,
        }

        sample.update(rdict)
        return sample


class ElasticTransform(MTTransform):
    "Elastic transform for 2D and 3D inputs"
    def __init__(self, alpha_range, sigma_range,
                 p=0.5, labeled=True):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.labeled = labeled
        self.p = p
        self.is3D = False

    @staticmethod
    def get_params(alpha, sigma):
        alpha = np.random.uniform(alpha[0], alpha[1])
        sigma = np.random.uniform(sigma[0], sigma[1])
        return alpha, sigma

    def elastic_transform(self, image, alpha, sigma):
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        if self.is3D:
            dz = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                                 sigma, mode="constant", cval=0) * alpha
            x, y, z = np.meshgrid(np.arange(shape[0]),
                                  np.arange(shape[1]),
                                  np.arange(shape[2]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))
        else:
            x, y = np.meshgrid(np.arange(shape[0]),
                               np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)

    def sample_augment(self, input_data, params):
        param_alpha, param_sigma = params
        np_input_data = np.array(input_data)
        if len(np_input_data.shape) == 3:
            self.is3D = True

        np_input_data = self.elastic_transform(np_input_data,
                                               param_alpha, param_sigma)
        if not self.is3D:
            input_data = Image.fromarray(np_input_data, mode='F')
        return input_data

    def label_augment(self, gt_data, params):
        param_alpha, param_sigma = params

        np_gt_data = np.array(gt_data)
        np_gt_data = self.elastic_transform(np_gt_data,
                                            param_alpha, param_sigma)
        if not self.is3D:
            np_gt_data[np_gt_data >= 0.5] = 255.0
            np_gt_data[np_gt_data < 0.5] = 0.0
            np_gt_data = np_gt_data.astype(np.uint8)
            gt_data = Image.fromarray(np_gt_data, mode='L')

        return gt_data

    def __call__(self, sample):
        rdict = {}

        if np.random.random() < self.p:
            input_data = sample['input']
            params = self.get_params(self.alpha_range,
                                     self.sigma_range)

            if isinstance(input_data, list):
                ret_input = [self.sample_augment(item, params)
                             for item in input_data]
            else:
                ret_input = self.sample_augment(input_data, params)
                
            rdict['input'] = ret_input

            if self.labeled:
                gt_data = sample['gt']
                if isinstance(gt_data, list):
                    ret_gt = [self.label_augment(item, params)
                              for item in gt_data]
                else:
                    ret_gt = self.label_augment(gt_data, params)

                rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


# TODO: Resample should keep state after changing state.
#       By changing pixel dimensions, we should be
#       able to return later to the original space.
class Resample(MTTransform):
    def __init__(self, wspace, hspace,
                 interpolation=Image.BILINEAR,
                 labeled=True):
        self.hspace = hspace
        self.wspace = wspace
        self.interpolation = interpolation
        self.labeled = labeled

    @staticmethod
    def resample_bin(self, data, wshape, hshape, thr=0.5):
        data = data.resize((wshape, hshape), resample=self.interpolation)
        np_data = np.array(data)
        np_data[np_data > thr] = 255.0
        np_data[np_data <= thr] = 0.0
        np_data = np_data.astype(np.uint8)
        data = Image.fromarray(np_data, mode='L')
        return data

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        input_metadata = sample['input_metadata'][0]

        # Voxel dimension in mm
        hzoom, wzoom = input_metadata["zooms"]
        hshape, wshape = input_metadata["data_shape"]

        hfactor = hzoom / self.hspace
        wfactor = wzoom / self.wspace

        hshape_new = int(hshape * hfactor)
        wshape_new = int(wshape * wfactor)

        if isinstance(input_data, list):
            ret_input = [item.resize((wshape_new, hshape_new), resample=self.interpolation) for item in input_data]
        else:
            ret_input = input_data.resize((wshape_new, hshape_new),
                                          resample=self.interpolation)
        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            rdict['gt'] = self.resample_bin(gt_data, wshape_new,
                                            hshape_new)
        if sample['roi'] is not None:
            roi_data = sample['roi']
            rdict['roi'] = self.resample_bin(roi_data, wshape_new,
                                             hshape_new, thr=0.0)

        sample.update(rdict)
        return sample


class AdditiveGaussianNoise(MTTransform):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        noise = np.random.normal(self.mean, self.std, input_data[0].size)
        noise = noise.astype(np.float32)
        noisy_input = []
        for item in input_data:
            np_input_data = np.array(item)
            np_input_data += noise
            noisy_input.append(Image.fromarray(np_input_data, mode='F'))
        rdict['input'] = noisy_input

        sample.update(rdict)
        return sample


class Clahe(MTTransform):
    def __init__(self, clip_limit=3.0, kernel_size=(8, 8)):
        # Default values are based upon the following paper:
        # https://arxiv.org/abs/1804.09400 (3D Consistent Cardiac Segmentation)

        self.clip_limit = clip_limit
        self.kernel_size = kernel_size

    def __call__(self, sample):
        if not isinstance(sample, np.ndarray):
            raise TypeError("Input sample must be a numpy array.")
        input_sample = np.copy(sample)
        array = skimage.exposure.equalize_adapthist(
            input_sample,
            kernel_size=self.kernel_size,
            clip_limit=self.clip_limit
        )
        return array


class HistogramClipping(MTTransform):
    def __init__(self, min_percentile=5.0, max_percentile=95.0):
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def __call__(self, sample):
        array = np.copy(sample)
        percentile1 = np.percentile(array, self.min_percentile)
        percentile2 = np.percentile(array, self.max_percentile)
        array[array <= percentile1] = percentile1
        array[array >= percentile2] = percentile2
        return array
