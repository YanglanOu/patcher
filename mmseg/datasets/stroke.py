from fileinput import filename
import os.path as osp
import tempfile
import cv2

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image
from mmseg.core import mean_iou

from .builder import DATASETS
from .custom import CustomDataset
from utils import dice_coeff, overlay


@DATASETS.register_module()
class StrokeDataset(CustomDataset):
    """Storke dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('nostroke', 'stroke')

    PALETTE = [[255, 255, 255], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(StrokeDataset, self).__init__(
            img_suffix='.npy',
            seg_map_suffix='.npy',
            **kwargs)

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            gt_seg_map = np.load(seg_map)
            # modify if custom classes
            if self.label_map is not None:
                for old_id, new_id in self.label_map.items():
                    gt_seg_map[gt_seg_map == old_id] = new_id
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 vis=False,
                 save_dir=None,
                 imgfile_prefix=None):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]

        # if len(metrics) > 0:
        #     eval_results.update(
        #         super(StrokeDataset,
        #               self).evaluate(results, metrics, logger))
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mIoU']
        if metric not in allowed_metrics:
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()

        if vis:
            for i in range(len(gt_seg_maps)):
                target = gt_seg_maps[i]
                preds = results[i]
                save_fig = overlay(target, preds)
                name = self.img_infos[i]['filename'].split('.')[0]
                save_name = f'{save_dir}/{name}.png'
                pred_name = f'{save_dir}/preds_{name}.png'
                target_name = f'{save_dir}/labels_{name}.png'
                cv2.imwrite(save_name, save_fig)
                # cv2.imwrite(pred_name, preds*255)
                # cv2.imwrite(target_name, target*255)
                # cv2.imwrite(save_name, preds)

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        all_acc, acc, iou = mean_iou(
            results, gt_seg_maps, num_classes, ignore_index=self.ignore_index)
        dice = dice_coeff(results, gt_seg_maps)
        summary_str = ''
        summary_str += 'per class results:\n'

        line_format = '{:<15} {:>10} {:>10}\n'
        summary_str += line_format.format('Class', 'IoU', 'Acc')
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        for i in range(num_classes):
            iou_str = '{:.2f}'.format(iou[i] * 100)
            acc_str = '{:.2f}'.format(acc[i] * 100)
            summary_str += line_format.format(class_names[i], iou_str, acc_str)
        summary_str += 'Summary:\n'
        line_format = '{:<15} {:>10} {:>10} {:>10} {:>10}\n'
        summary_str += line_format.format('Scope', 'mIoU', 'dice', 'mAcc', 'aAcc')

        iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
        acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
        all_acc_str = '{:.2f}'.format(all_acc * 100)
        dice_str = '{:.2f}'.format(dice * 100)
        summary_str += line_format.format('global', iou_str, dice_str, acc_str,
                                          all_acc_str)
        print_log(summary_str, logger)

        eval_results['mIoU'] = np.nanmean(iou)
        eval_results['dice'] = dice
        eval_results['mAcc'] = np.nanmean(acc)
        eval_results['aAcc'] = all_acc
        eval_results['stroke_IoU'] = iou[1]

        return eval_results
        # return eval_results

