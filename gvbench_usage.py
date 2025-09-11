import os
import numpy as np
import argparse
import torch
import subprocess
from tqdm import tqdm
from scipy.special import softmax
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from utils.process_database import create_image_pair_list, remove_doppelgangers
from jw_utils.jw_utils.geometric_verification import eval


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Test GV-Bench with Doppelgangers classification model.')
    parser.add_argument('--data_root', type=str, required=True, help='Path to the input image dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output results.')
    parser.add_argument('--pretrained', type=str, default='checkpoints/dopp-crop-focalloss_lr1e-3_warmup20/checkpoint-best.pth', help="Path to the pretrained model checkpoint.")    
    args = parser.parse_args()
    return args
        
def doppelgangers_classifier(args):
    """Classify image pairs to filter Doppelgangers."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # read this function for model configuration
    model = AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', head_type_dg='transformer', 
                             output_mode='pts3d+desc24', output_mode_dg='dg_score', depth_mode=('exp', -np.inf, np.inf), conf_mode=('exp', 1, np.inf), 
                             enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, np.inf), 
                             add_dg_pred_head=True, freeze=['mask','encoder','decoder','head']).from_pretrained(args.pretrained).to(device)

    # pairs = np.load(f"{args.output_path}/pairs_list.npy")
    pairs = np.load(f"{args.data_root}/labels/day.npy", allow_pickle=True)
    
    prob_list = []
    gt_list = []

    for pair in tqdm(pairs, desc="Disambiguating pairs"):
        img1, img2, gt = pair
        gt_list.append(gt)
        img_paths = [os.path.join(args.data_root, "images", img) for img in [img1, img2]]
        images = load_images(img_paths, size=512, verbose=False)
        output = inference(make_pairs(images), model, device, verbose=False)

        pred1, pred2 = output['pred1'], output['pred2']
        if isinstance(output['pred1'], list):
            pred1 = torch.stack(output['pred1'], dim=0)
        else:
            pred1 = output['pred1']
            
        if isinstance(output['pred2'], list):
            pred2 = torch.stack(output['pred2'], dim=0)
        else:
            pred2 = output['pred2']
            
        score_s1 = softmax(pred1.detach().cpu().numpy(), axis=1)
        score_s2 = softmax(pred2.detach().cpu().numpy(), axis=1)
        vote_0 = sum(score_s1[:,0] > score_s1[:,1]) + sum(score_s2[:,0] > score_s2[:,1])
        vote_1 = sum(score_s1[:,1] > score_s1[:,0]) + sum(score_s2[:,1] > score_s2[:,0])
        if vote_1 > vote_0:
            score = np.max((score_s1[:,1], score_s2[:,1]))
        elif vote_1 < vote_0:
            score = np.min((score_s1[:,1], score_s2[:,1]))
        else:
            score = np.mean((score_s1[:,1], score_s2[:,1]))
            
        prob_list.append(score)

    np.save(f"{args.output_path}/pair_probability_list_dgpp.npy", {'prob': np.array(prob_list).reshape(-1, 1), 'label': np.array(gt_list).reshape(-1, 1)})
    
    return prob_list, gt_list

def main():
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    prob_list, gt_list = doppelgangers_classifier(args)
    mr = eval.max_recall(np.array(prob_list), np.array(gt_list))
    eval.plot_pr_curve(np.array(prob_list), np.array(gt_list))
    import matplotlib.pyplot as plt
    plt.show()
    
if __name__ == '__main__':
    main()
