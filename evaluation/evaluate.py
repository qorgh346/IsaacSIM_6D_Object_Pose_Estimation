import os
import torch
import random
from network.IS_OPE import ISOPENetwork
from tools.geom_utils import generate_RT, generate_sRT
from config.config import *
from absl import app

FLAGS = flags.FLAGS
from evaluation.load_data_eval import PoseDataset

# from datasets.load_data import PoseDataset

import torch.nn as nn
import numpy as np
import time

# from creating log
import tensorflow as tf
import evaluation
from evaluation.eval_utils import setup_logger, compute_mAP
from evaluation.eval_utils_v1 import compute_degree_cm_mAP
from tqdm import tqdm

device = 'cuda'


def evaluate(argv):
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    tf.compat.v1.disable_eager_execution()
    logger = setup_logger('eval_log', os.path.join(FLAGS.model_save, 'log_eval.txt'))
    Train_stage = 'PoseNet_only'
    FLAGS.train = False
    model_name = os.path.basename(FLAGS.resume_model).split('.')[0]
    val_dataset = PoseDataset(source=FLAGS.dataset, mode='test')
    output_path = os.path.join(FLAGS.model_save, f'eval_result_{model_name}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    import pickle
    pred_result_save_path = os.path.join(output_path, 'pred_result.pkl')
    if os.path.exists(pred_result_save_path):
        with open(pred_result_save_path, 'rb') as file:
            pred_results = pickle.load(file)
    else:
        network = ISOPENetwork(Train_stage)
        network = network.to(device)
        if FLAGS.resume:
            state_dict = torch.load(FLAGS.resume_model)
            network.load_state_dict(state_dict)
        else:
            raise NotImplementedError
        # start to test
        # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle= False,
        #                                                num_workers=1)
        network = network.eval()
        pred_results = []

        for i, data in tqdm(enumerate(val_dataset, 1)):
            if data is None:
                continue
            data, detection_dict, gts , img_path = data
            mean_shape = data['mean_shape'].to(device)
            sym = data['sym_info'].to(device)
            # cat_id
            # if len(data['cat_id']) == 0:
            #     detection_dict['pred_RTs'] = np.zeros((0, 4, 4))
            #     detection_dict['pred_scales'] = np.zeros((0, 4, 4))
            #     pred_results.append(detection_dict)
            #     continue
            output_dict \
                = network(rgb=data['roi_img'].to(device), depth=data['roi_depth'].to(device),
                          depth_normalize=data['depth_normalize'].to(device),
                          obj_id=data['cat_id'].to(device), camK=data['cam_K'].to(device),
                          gt_mask=data['roi_mask'].to(device),
                          gt_R=None, gt_t=None, gt_s=None, mean_shape=mean_shape,
                          gt_2D=data['roi_coord_2d'].to(device), sym=sym,
                          def_mask=data['roi_mask'].to(device))

            p_green_R_vec = output_dict['p_green_R'].detach()
            p_red_R_vec = output_dict['p_red_R'].detach()
            p_T = output_dict['Pred_T'].detach()
            p_s = output_dict['Pred_s'].detach()
            f_green_R = output_dict['f_green_R'].detach()
            f_red_R = output_dict['f_red_R'].detach()
            from tools.training_utils import get_gt_v
            pred_s = p_s + mean_shape
            pred_RT,pred_r = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)
            #
            # gt_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)
            time.sleep(1)
            print('\n')
            print('img_path = ',img_path)

            print('pred_translation = ',p_T)
            print('pred_rotation    =',pred_RT)

            print('gt translation   = ',data['gt_trans'].detach())
            print('gt_rotation      = ',data['gt_rot'].detach())

            print('model_name = ',gts['model_list'])

            if pred_RT is not None:
                pred_RT = pred_RT.detach().cpu().numpy()
                pred_s = pred_s.detach().cpu().numpy()
                detection_dict['pred_RTs'] = pred_RT
                detection_dict['pred_scales'] = pred_s
                # detection_dict['gt_RTs'] = output_dict['']
            else:
                assert NotImplementedError
            pred_results.append(detection_dict)
        with open(pred_result_save_path, 'wb') as file:
            pickle.dump(pred_results, file)

    if FLAGS.eval_inference_only:
        import sys
        sys.exit()

    categoryNum = 2
    degree_thres_list = list(range(0, categoryNum, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]


    #iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, output_path, degree_thres_list, shift_thres_list,
    #                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)
    synset_names = ['BG'] + ['can','box'] #['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1
    iou_aps, pose_aps = compute_degree_cm_mAP(pred_results, synset_names, output_path, degree_thres_list, shift_thres_list,
                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)

    # # fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    categoryNum = 2
    degree_thres_list = list(range(0, categoryNum, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]


    #iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, output_path, degree_thres_list, shift_thres_list,
    #                                                  iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)
    synset_names = ['BG'] + ['can','box'] #['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    if FLAGS.per_obj in synset_names:
        idx = synset_names.index(FLAGS.per_obj)
    else:
        idx = -1
    iou_aps, pose_aps = compute_degree_cm_mAP(pred_results, synset_names, output_path, degree_thres_list, shift_thres_list,
                              iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True,)

    # # fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    shift_10_idx = shift_thres_list.index(10)

    messages = []

    if FLAGS.per_obj in synset_names:
        messages.append('mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))
    else:
        messages.append('average mAP:')
        messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
        messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
        messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
        messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
        messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
        messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
        messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
        messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))

        for idx in range(1, len(synset_names)):
            messages.append('category {}'.format(synset_names[idx]))
            messages.append('mAP:')
            messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[idx, iou_25_idx] * 100))
            messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[idx, iou_50_idx] * 100))
            messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[idx, iou_75_idx] * 100))
            messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_02_idx] * 100))
            messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_05_idx, shift_05_idx] * 100))
            messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_02_idx] * 100))
            messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_05_idx] * 100))
            messages.append('10 degree, 10cm: {:.1f}'.format(pose_aps[idx, degree_10_idx, shift_10_idx] * 100))

    for msg in messages:
        logger.info(msg)

if __name__ == "__main__":
    app.run(evaluate)