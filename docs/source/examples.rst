Examples
===============================================================================
In this section you can see various examples using MedicalTorch API.

U-Net with GM Segmentation Challenge
--------------------------------------------------------------------------------
Please note that this example requires
`TensorboardX <https://github.com/lanpa/tensorboardX>`_ to write statistics
into a TensorBoard format. You can install it with::

    pip install tensorboardx

The example is described below:

.. literalinclude:: ../../examples/gmchallenge_unet.py
   :language: python

Dataloader Tutorial for NIFTI images
--------------------------------------------------------------------------------
The tutorial for creating a dataloader using medicaltorch can be found `here <https://github.com/perone/medicaltorch/tree/master/examples/Dataloaders_NIFTI.ipynb>`_ . 
