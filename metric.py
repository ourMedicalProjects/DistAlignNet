import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.box_utils import iou_3D, center_distance
from config import data_config
from statistics import mean
from tqdm import tqdm

def load(root_data_path, test_id, post):
    return np.load(os.path.join(root_data_path, test_id + post), allow_pickle=True)
    
fold_cnt = 9
root_data_path = data_config['data_dir']
OUTPUT_DIR = f"error/{fold_cnt}"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def calc_geometry_metric(id_list, pbb_data_dir):
    max_iou_list = []
    min_distance_list = []
    
    for id in tqdm(id_list):
        if len(id) < 3:
            continue
        
        pbbs = load(pbb_data_dir, id, "_pbb.npy")
        ground_truth = load(root_data_path, id, "_bboxes.npy")
        
        for gt in ground_truth:
            max_iou = -1
            min_distance = 1000000

            for pbb in pbbs:
                max_iou = max(max_iou, iou_3D(pbb[1:], gt))
                min_distance = min(min_distance, center_distance(pbb[1:], gt))
            if max_iou > 0.05:
                max_iou_list.append(max_iou)
                min_distance_list.append(min_distance)
    
    
    print(f"MIOU is {mean(max_iou_list)}!")    
    print(f"MDIS is {mean(min_distance_list)}")
    
    return 

def print_det_result(id_list, pbb_data_dir):
    for id in id_list:
        if len(id) < 3:
            continue        
        
        print(f"Current process id is {id}")
        pbbs = load(pbb_data_dir, id, "_pbb.npy")
        ground_truth = load(root_data_path, id, "_bboxes.npy")
        img = load(root_data_path, id, ".npy")
    
        t_tier = 0
        tp_iter = 0
        
        for box in ground_truth:
            fig, ax = plt.subplots()
            ax.set_axis_off()
            cz, cy, cx, d = box
            slice_data = img[0][int(cz)]
            plt.imshow(slice_data, cmap='gray')
            rect = patches.Rectangle((cx - d, cy - d), d * 2, d * 2, linewidth=1, edgecolor='red' ,facecolor='none')
            ax.add_patch(rect)
            plt.savefig(os.path.join(OUTPUT_DIR, f"{id}_gt_{t_tier}.png"), bbox_inches=0)
            t_tier += 1
            plt.close(fig)
       
        _, depth, height, width = img.shape
        
        # for pbb in pbbs:
        #     p, cz, cy, cx, d = pbb
        #     if cz >= depth or cz < 0: 
        #         continue
            
        #     slice_data = img[0][int(cz)]
        #     p, cz, cy, cx, d = pbb
        #     if tp_iter > 3:
        #         break
        #     fig, ax = plt.subplots()
        #     ax.set_axis_off()
        #     plt.imshow(slice_data, cmap='gray')
        #     rect = patches.Rectangle((cx - d, cy - d), d * 2, d * 2, linewidth=1, edgecolor='red' ,facecolor='none')
        #     ax.add_patch(rect)
        #     # ax.text(cx + 5, cy + 5, f"{p:.3f}", color='red')
        #     plt.savefig(os.path.join(OUTPUT_DIR, f"{id}_tp_{tp_iter}.png"), bbox_inches=0)
        #     tp_iter += 1
        #     plt.close(fig)

    return 

if __name__ == "__main__":
    # 
    pbb_data_dir = f"experiment/coord/fold{fold_cnt}/res"
    pbb_data_list = os.listdir(pbb_data_dir)
    id_list = [id[:-8] for id in pbb_data_list]
    
    # print_det_result(id_list, pbb_data_dir)
    calc_geometry_metric(id_list, pbb_data_dir)
    # print_det_result(id_list, pbb_data_dir)