# -*-coding = utf-8 -*-
# @time : 2021/12/10 14:43
# @Author: 自在清风
# @File ： Evaluation.py
# @Software ：PyCharm
import glob
import cv2
import numpy as np
import os
import pandas as pd


def Evaluation(truth_path, predict_path, save):
    """Ensemble Dice as used in Computational Precision Medicine Challenge."""
    truth_paths = glob.glob(truth_path)
    predict_paths = glob.glob(predict_path)
    output = []
    for true, pred in zip(truth_paths, predict_paths):
        name = os.path.basename(true)
        true = cv2.imread(true, 0)
        pred = cv2.imread(pred, 0)
        true_id = list(np.unique(true))
        pred_id = list(np.unique(pred))
        # remove background aka id 0
        true_id.remove(0)
        pred_id.remove(0)

        total_markup = 0.001
        total_intersect = 0
        t_mask = 0
        p_mask = 0
        for t in true_id:
            t_mask = np.array(true == t, np.uint8)
            for p in pred_id:
                p_mask = np.array(pred == p, np.uint8)
                intersect = p_mask * t_mask
                if intersect.sum() > 0:
                    total_intersect = intersect.sum()
                    total_markup = t_mask.sum() + p_mask.sum()
        Dice = 2 * total_intersect / total_markup
        Recall = total_intersect / (t_mask.sum())
        Precision = total_intersect / (p_mask.sum() + 0.0001)
        tmp = [name, Recall, Precision, Dice]
        output.append(tmp)

    f = pd.DataFrame(output)
    f.to_csv(save, header=['name', 'R', 'P', 'Dice'], float_format='%.4f')


if __name__ == '__main__':
    path = ['fusion', 'hd', 'ms', 'unetpp', 'unet3p']
    truth = r'data/test/test_mask/*.png'  # truth label file path
    predict = r'data/detected_result/**/*.png'  # predict result file path
    save_path = r'data/Evaluation_result/**_evaluation.csv'  # evaluation result file path
    for p in path:
        save_path1 = save_path.replace('**', p)
        predict1 = predict.replace('**', p)
        Evaluation(truth_path=truth, predict_path=predict1, save=save_path1)
    print('done!')
