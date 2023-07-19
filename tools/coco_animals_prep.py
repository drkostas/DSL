import json
import os
import argparse
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy


def convert(args, seed=42):
    print("Creating base folders")
    out_annotations = os.path.join(args.output, 'annotations')
    out_train2017 = os.path.join(args.output, 'train2017')
    out_val2017 = os.path.join(args.output, 'val2017')
    # out_unlabeled2017 = os.path.join(args.output, 'unlabeled2017')
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    if not os.path.exists(out_annotations):
        os.mkdir(out_annotations)
    if not os.path.exists(out_train2017):
        os.mkdir(out_train2017)
    if not os.path.exists(out_val2017):
        os.mkdir(out_val2017)
    # if not os.path.exists(out_unlabeled2017):
    #     os.mkdir(out_unlabeled2017)

    print("Loading annotations")
    with open(os.path.join(args.input, 'annotations/instances_train2017.json')) as f:
        in_annos = json.load(f)
        in_images = in_annos['images']
        in_annotations = in_annos['annotations']
    
    # Keep only images that exist in train2017
    print(f"Originally have {len(in_images)} images")
    print("Filtering images")
    in_images_exist = deepcopy(in_images)
    for image in in_images:
        if not os.path.exists(os.path.join(args.input,
                                           'train2017',
                                           image['file_name'])):
            in_images_exist.remove(image)
    in_images = in_images_exist
    print(f"Filtered to {len(in_images)} images")
    
    in_idx = [item['id'] for item in in_images]
    in_annotations = [item for item in in_annotations if item['image_id'] in in_idx]
    dummy_y_in = np.zeros(len(in_idx))
    
    train_idx, val_idx, _, _ = train_test_split(in_idx,
                                                dummy_y_in,
                                                train_size=(1.0 - args.val_perc),
                                                shuffle=True,
                                                random_state=seed)
    print(f"Train: {len(train_idx)} images")
    print(f"Val: {len(val_idx)} images")
    
    # dummy_y_train = np.zeros(len(train_idx))
    # train_idx, unlabeled_idx, _, _ = train_test_split(train_idx,
    #                                                   dummy_y_train,
    #                                                   train_size=(1.0 - args.labeled_perc),
    #                                                   shuffle=True,
    #                                                   random_state=seed)
    print(f"Final Train: {len(train_idx)} images")
    print(f"Final Val: {len(val_idx)} images")
    # print(f"Final Unlabeled: {len(unlabeled_idx)} images")
    
    print("Creating annotation jsons")
    #! Train
    out_train_annos = deepcopy(in_annos)
    annos_annotations = []
    annos_images = []
    for image in out_train_annos['images']:
        if image['id'] in train_idx:
            annos_images.append(image)
            annotations_matched = [item for item in out_train_annos['annotations']
                                   if item['image_id'] == image['id']]
            annos_annotations.extend(annotations_matched)
    out_train_annos['images'] = annos_images
    out_train_annos['annotations'] = annos_annotations
    with open(os.path.join(out_annotations, 'instances_train2017.json'), 'w') as f:
        json.dump(out_train_annos, f)
    #! Val
    out_val_annos = deepcopy(in_annos)
    annos_annotations = []
    annos_images = []
    for image in out_val_annos['images']:
        if image['id'] in val_idx:
            annos_images.append(image)
            annotations_matched = [item for item in out_val_annos['annotations']
                                   if item['image_id'] == image['id']]
            annos_annotations.extend(annotations_matched)
    out_val_annos['images'] = annos_images
    out_val_annos['annotations'] = annos_annotations
    with open(os.path.join(out_annotations, 'instances_val2017.json'), 'w') as f:
        json.dump(out_val_annos, f)
    #! Unlabeled
    # out_unlabeled_annos = deepcopy(in_annos)
    # annos_annotations = []
    # annos_images = []
    # for image in out_unlabeled_annos['images']:
    #     if image['id'] in unlabeled_idx:
    #         annos_images.append(image)
    #         annotations_matched = [item for item in out_unlabeled_annos['annotations']
    #                                if item['image_id'] == image['id']]
    #         annos_annotations.extend(annotations_matched)
    # out_unlabeled_annos['images'] = annos_images
    # out_unlabeled_annos['annotations'] = annos_annotations
    # with open(os.path.join(out_annotations, 'instances_unlabeled2017.json'), 'w') as f:
    #     json.dump(out_unlabeled_annos, f)
    
    # Create train2017 folder
    print("Copying images to train2017, val2017, unlabeled2017 folders")
    for image in in_images:
        if image['id'] in train_idx:
            shutil.copy(os.path.join(args.input, 'train2017', image['file_name']), out_train2017)
        elif image['id'] in val_idx:
            shutil.copy(os.path.join(args.input, 'train2017', image['file_name']), out_val2017)
        # elif image['id'] in unlabeled_idx:
        #     shutil.copy(os.path.join(args.input, 'train2017', image['file_name']), out_unlabeled2017)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting input path and output path")
    parser.add_argument('--input', type=str, default="../COCO_ANIMALS", help="coco dir path: ../COCO_ANIMALS")
    parser.add_argument('--output', type=str, default="../COCO_ANIMALS2", help="coco dir path: ../COCO_ANIMALS2")
    parser.add_argument('--labeled_perc', type=float, default=0.1)
    parser.add_argument('--val_perc', type=float, default=0.07765803)
    args = parser.parse_args()
    # args.output = f"{args.output}_lab{args.labeled_perc:.2f}_val{args.val_perc:.2f}"
    args.output = f"{args.output}_val{args.val_perc:.2f}"
    convert(args)
