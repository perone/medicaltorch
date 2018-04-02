import torch
from torch.autograd import Variable

from medicaltorch import models as mt_models


class TestModels(object):
    def test_aspp(self):
        model = mt_models.NoPoolASPP()
        random_data = torch.randn(1, 1, 200, 200)
        random_var = Variable(random_data)
        output = model(random_var)
        assert output.size() == (1, 1, 200, 200)
