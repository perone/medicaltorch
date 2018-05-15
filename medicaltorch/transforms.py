import numpy as np
import numbers
import torchvision.transforms.functional as F
from torchvision import transforms
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
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        if isinstance(input_data, list):
            ret_input = [F.to_tensor(item)
                         for item in input_data]
        else:
            ret_input = F.to_tensor(input_data)

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                if isinstance(gt_data, list):
                    ret_gt = [F.to_tensor(item)
                              for item in gt_data]
                else:
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
            ret_input = [self.sample_transform(item)
                         for item in input_data]
        else:
            ret_input = self.sample_transform(input_data)

        rdict['input'] = ret_input

        if self.labeled:
            gt_data = sample['gt']

            if isinstance(gt_data, list):
                ret_gt = [self.sample_transform(item)
                          for item in gt_data]
            else:
                ret_gt = self.sample_transform(gt_data)

            rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample


class UnCenterCrop2D(MTTransform):
    def __init__(self, size, segmentation=True):
        self.size = size
        self.segmentation = segmentation

    def __call__(self, sample):
        input_data, gt_data = sample['input'], sample['gt']
        input_metadata, gt_metadata = sample['input_metadata'], sample['gt_metadata']

        (fh, fw, w, h) = input_metadata["__centercrop"]
        (fh, fw, w, h) = gt_metadata["__centercrop"]

        return sample


class CenterCrop2D(MTTransform):
    """Make a center crop of a specified size.

    :param segmentation: if it is a segmentation task.
                         When this is True (default), the crop
                         will also be applied to the ground truth.
    """
    def __init__(self, size, labeled=True):
        self.size = size
        self.labeled = labeled

    @staticmethod
    def propagate_params(sample, params):
        input_metadata = sample['input_metadata']
        input_metadata["__centercrop"] = params
        return input_metadata

    @staticmethod
    def get_params(sample):
        input_metadata = sample['input_metadata']
        return input_metadata["__centercrop"]

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        w, h = input_data.size
        th, tw = self.size
        fh = int(round((h - th) / 2.))
        fw = int(round((w - tw) / 2.))

        params = (fh, fw, w, h)
        self.propagate_params(sample, params)

        input_data = F.center_crop(input_data, self.size)
        rdict['input'] = input_data

        if self.labeled:
            gt_data = sample['gt']
            gt_metadata = sample['gt_metadata']
            gt_data = F.center_crop(gt_data, self.size)
            gt_metadata["__centercrop"] = (fh, fw, w, h)
            rdict['gt'] = gt_data


        sample.update(rdict)
        return sample

    def undo_transform(self, sample):
        rdict = {}
        input_data = sample['input']
        fh, fw, w, h = self.get_params(sample)
        th, tw = self.size

        pad_left = fw
        pad_right = w - pad_left - tw
        pad_top = fh
        pad_bottom = h - pad_top - th

        padding = (pad_left, pad_top, pad_right, pad_bottom)
        input_data = F.pad(input_data, padding)
        rdict['input'] = input_data

        sample.update(rdict)
        return sample


class Normalize(MTTransform):
    """Normalize a tensor image with mean and standard deviation.
    
    :param mean: mean value.
    :param std: standard deviation value.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        input_data = sample['input']

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

        mean, std = input_data.mean(), input_data.std()
        input_data = F.normalize(input_data, [mean], [std])

        rdict = {
            'input': input_data,
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
        input_data = sample['input']
        angle = self.get_params(self.degrees)
        input_data = F.rotate(input_data, angle,
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
        np_gt_data[np_gt_data >= 0.5] = 1.0
        np_gt_data[np_gt_data < 0.5] = 0.0
        gt_data = Image.fromarray(np_gt_data, mode='F')
        return gt_data

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.
        Returns:
            PIL Image: Affine transformed image.
        """
        rdict = {}
        input_data = sample['input']

        if isinstance(input_data, list):
            input_data_size = input_data[0].size
        else:
            input_data_size = input_data.size

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
            #ret_input = [self.sample_augment(item, params)
            #             for item in input_data]

            # Augment just the image, not the mask
            # TODO: fix it later
            ret_input = []
            ret_input.append(self.sample_augment(input_data[0], params))
            ret_input.append(input_data[1])
        else:
            ret_input = self.sample_augment(input_data, params)

        rdict = {
            'input': ret_input,
        }

        sample.update(rdict)
        return sample


class ElasticTransform(MTTransform):
    def __init__(self, alpha_range, sigma_range,
                 p=0.5, labeled=True):
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range
        self.labeled = labeled
        self.p = p

    @staticmethod
    def get_params(alpha, sigma):
        alpha = np.random.uniform(alpha[0], alpha[1])
        sigma = np.random.uniform(sigma[0], sigma[1])
        return alpha, sigma

    @staticmethod
    def elastic_transform(image, alpha, sigma):
        shape = image.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1),
                             sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]),
                           np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
        return map_coordinates(image, indices, order=1).reshape(shape)

    def sample_augment(self, input_data, params):
        param_alpha, param_sigma = params

        np_input_data = np.array(input_data)
        np_input_data = self.elastic_transform(np_input_data,
                                               param_alpha, param_sigma)
        input_data = Image.fromarray(np_input_data, mode='F')
        return input_data

    def label_augment(self, gt_data, params):
        param_alpha, param_sigma = params

        np_gt_data = np.array(gt_data)
        np_gt_data = self.elastic_transform(np_gt_data,
                                            param_alpha, param_sigma)
        np_gt_data[np_gt_data >= 0.5] = 1.0
        np_gt_data[np_gt_data < 0.5] = 0.0
        gt_data = Image.fromarray(np_gt_data, mode='F')

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

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        input_metadata = sample['input_metadata']

        # Voxel dimension in mm
        hzoom, wzoom = input_metadata["zooms"]
        hshape, wshape = input_metadata["data_shape"]

        hfactor = hzoom / self.hspace
        wfactor = wzoom / self.wspace

        hshape_new = int(hshape * hfactor)
        wshape_new = int(wshape * wfactor)

        input_data = input_data.resize((wshape_new, hshape_new),
                                       resample=self.interpolation)
        rdict['input'] = input_data

        if self.labeled:
            gt_data = sample['gt']
            gt_metadata = sample['gt_metadata']
            gt_data = gt_data.resize((wshape_new, hshape_new),
                                     resample=self.interpolation)
            np_gt_data = np.array(gt_data)
            np_gt_data[np_gt_data >= 0.5] = 1.0
            np_gt_data[np_gt_data < 0.5] = 0.0
            gt_data = Image.fromarray(np_gt_data, mode='F')
            rdict['gt'] = gt_data

        sample.update(rdict)
        return sample


class AdditiveGaussianNoise(MTTransform):
    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        noise = np.random.normal(self.mean, self.std, input_data.size)
        noise = noise.astype(np.float32)

        np_input_data = np.array(input_data)
        np_input_data += noise
        input_data = Image.fromarray(np_input_data, mode='F')
        rdict['input'] = input_data

        sample.update(rdict)
        return sample
