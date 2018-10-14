from collections import defaultdict
import time
import os

import numpy as np

from tqdm import tqdm

from tensorboardX import SummaryWriter

from medicaltorch import datasets as mt_datasets
from medicaltorch import models as mt_models
from medicaltorch import transforms as mt_transforms
from medicaltorch import losses as mt_losses
from medicaltorch import metrics as mt_metrics
from medicaltorch import filters as mt_filters

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torchvision.utils as vutils

cudnn.benchmark = True


def threshold_predictions(predictions, thr=0.999):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


def run_main():
    train_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((200, 200)),
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    val_transform = transforms.Compose([
        mt_transforms.CenterCrop2D((200, 200)),
        mt_transforms.ToTensor(),
        mt_transforms.NormalizeInstance(),
    ])

    # Here we assume that the SC GM Challenge data is inside the folder
    # "../data" and it was previously resampled.
    gmdataset_train = mt_datasets.SCGMChallenge2DTrain(root_dir="../data",
                                                       subj_ids=range(1, 9),
                                                       transform=train_transform,
                                                       slice_filter_fn=mt_filters.SliceFilter())

    # Here we assume that the SC GM Challenge data is inside the folder
    # "../data" and it was previously resampled.
    gmdataset_val = mt_datasets.SCGMChallenge2DTrain(root_dir="../data",
                                                     subj_ids=range(9, 11),
                                                     transform=val_transform)

    train_loader = DataLoader(gmdataset_train, batch_size=16,
                              shuffle=True, pin_memory=True,
                              collate_fn=mt_datasets.mt_collate,
                              num_workers=1)

    val_loader = DataLoader(gmdataset_val, batch_size=16,
                            shuffle=True, pin_memory=True,
                            collate_fn=mt_datasets.mt_collate,
                            num_workers=1)

    model = mt_models.Unet(drop_rate=0.4, bn_momentum=0.1)
    model.cuda()

    num_epochs = 200
    initial_lr = 0.001

    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    writer = SummaryWriter(log_dir="log_exp")
    for epoch in tqdm(range(1, num_epochs+1)):
        start_time = time.time()

        scheduler.step()

        lr = scheduler.get_lr()[0]
        writer.add_scalar('learning_rate', lr, epoch)

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        for i, batch in enumerate(train_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]

            var_input = input_samples.cuda()
            var_gt = gt_samples.cuda(non_blocking=True)

            preds = model(var_input)

            loss = mt_losses.dice_loss(preds, var_gt)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            if epoch % 5 == 0:
                grid_img = vutils.make_grid(input_samples,
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Input', grid_img, epoch)

                grid_img = vutils.make_grid(preds.data.cpu(),
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Predictions', grid_img, epoch)

                grid_img = vutils.make_grid(gt_samples,
                                            normalize=True,
                                            scale_each=True)
                writer.add_image('Ground Truth', grid_img, epoch)

        train_loss_total_avg = train_loss_total / num_steps

        model.eval()
        val_loss_total = 0.0
        num_steps = 0

        metric_fns = [mt_metrics.dice_score,
                      mt_metrics.hausdorff_score,
                      mt_metrics.precision_score,
                      mt_metrics.recall_score,
                      mt_metrics.specificity_score,
                      mt_metrics.intersection_over_union,
                      mt_metrics.accuracy_score]

        metric_mgr = mt_metrics.MetricManager(metric_fns)

        for i, batch in enumerate(val_loader):
            input_samples, gt_samples = batch["input"], batch["gt"]

            with torch.no_grad():
                var_input = input_samples.cuda()
                var_gt = gt_samples.cuda(async=True)

                preds = model(var_input)
                loss = mt_losses.dice_loss(preds, var_gt)
                val_loss_total += loss.item()

            # Metrics computation
            gt_npy = gt_samples.numpy().astype(np.uint8)
            gt_npy = gt_npy.squeeze(axis=1)

            preds = preds.data.cpu().numpy()
            preds = threshold_predictions(preds)
            preds = preds.astype(np.uint8)
            preds = preds.squeeze(axis=1)

            metric_mgr(preds, gt_npy)

            num_steps += 1

        metrics_dict = metric_mgr.get_results()
        metric_mgr.reset()

        writer.add_scalars('metrics', metrics_dict, epoch)

        val_loss_total_avg = val_loss_total / num_steps

        writer.add_scalars('losses', {
                                'val_loss': val_loss_total_avg,
                                'train_loss': train_loss_total_avg
                            }, epoch)

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

        writer.add_scalars('losses', {
                                'train_loss': train_loss_total_avg
                            }, epoch)


if __name__ == '__main__':
    run_main()
