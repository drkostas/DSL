import json
import os

base_dir = '../COCO_ANIMALS2_val0.08'

out_annotations = os.path.join(base_dir, 'annotations')
out_train2017 = os.path.join(base_dir, 'train2017')
images_in_train2017 = os.listdir(out_train2017)
out_val2017 = os.path.join(base_dir, 'val2017')
images_in_val2017 = os.listdir(out_val2017)
# out_unlabeled2017 = os.path.join(base_dir, 'unlabeled2017')
# images_in_unlabeled2017 = os.listdir(out_unlabeled2017)

with open(os.path.join(out_annotations, 'instances_train2017.json'), 'r') as f:
    train_annos = json.load(f)

with open(os.path.join(out_annotations, 'instances_val2017.json'), 'r') as f:
    val_annos = json.load(f)

# with open(os.path.join(out_annotations, 'instances_unlabeled2017.json'), 'r') as f:
#     unlabeled_annos = json.load(f)
    
# Check if all images are in the annotations
print("Checking if all image files exist in jsons")

#! Train
print("Train")
num_found = 0
num_not_found = 0
for image in train_annos['images']:
    json_image = image['file_name']
    if json_image not in images_in_train2017:
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} and {num_not_found} not found")

#! Val
print("Val")
num_found = 0
num_not_found = 0
for image in val_annos['images']:
    json_image = image['file_name']
    if json_image not in images_in_val2017:
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} and {num_not_found} not found")

#! Unlabeled
# print("Unlabeled")
# num_found = 0
# num_not_found = 0
# for image in unlabeled_annos['images']:
#     json_image = image['file_name']
#     if json_image not in images_in_unlabeled2017:
#         num_not_found += 1
#     else:
#         num_found += 1
# print(F"Found {num_found} and {num_not_found} not found")

# Check if all annotations are in the images
print("Checking if all images in jsons exist as files")

#! Train
print("Train")
num_found = 0
num_not_found = 0
for image in images_in_train2017:
    json_image = image
    if json_image not in [item['file_name'] for item in train_annos['images']]:
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} and {num_not_found} not found")

#! Val
print("Val")
num_found = 0
num_not_found = 0
for image in images_in_val2017:
    json_image = image
    if json_image not in [item['file_name'] for item in val_annos['images']]:
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} and {num_not_found} not found")

#! Unlabeled
# print("Unlabeled")
# num_found = 0
# num_not_found = 0
# for image in images_in_unlabeled2017:
#     json_image = image
#     if json_image not in [item['file_name'] for item in unlabeled_annos['images']]:
#         num_not_found += 1
#     else:
#         num_found += 1
# print(F"Found {num_found} and {num_not_found} not found")

# Check if all annotations have corresponding image references (all in jsons)
print("Checking if annotations in jsons have corresponding image references in the same jsons")

#! Train
print("Train")
num_found = 0
num_not_found = 0
for annotation in train_annos['annotations']:
    if annotation['image_id'] not in [item['id'] for item in train_annos['images']]:
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} and {num_not_found} not found")

#! Val
print("Val")
num_found = 0
num_not_found = 0
for annotation in val_annos['annotations']:
    if annotation['image_id'] not in [item['id'] for item in val_annos['images']]:
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} and {num_not_found} not found")

#! Unlabeled
# print("Unlabeled")
# num_found = 0
# num_not_found = 0
# for annotation in unlabeled_annos['annotations']:
#     if annotation['image_id'] not in [item['id'] for item in unlabeled_annos['images']]:
#         num_not_found += 1
#     else:
#         num_found += 1
# print(F"Found {num_found} and {num_not_found} not found")


# Check if all images have corresponding annotation references (all in jsons)
print("Checking if images in jsons have corresponding annotations in the same jsons")

#! Train
print("Train")
num_found = 0
num_not_found = 0
for image in train_annos['images']:
    if image['id'] not in [annot['image_id'] for annot in train_annos['annotations']]:
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} and {num_not_found} not found")

#! Val
print("Val")
num_found = 0
num_not_found = 0
for image in val_annos['images']:
    if image['id'] not in [annot['image_id'] for annot in val_annos['annotations']]:
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} and {num_not_found} not found")


#! Unlabeled
# print("Unlabeled")
# num_found = 0
# num_not_found = 0
# for image in unlabeled_annos['images']:
#     if image['id'] not in [annot['image_id'] for annot in unlabeled_annos['annotations']]:
#         num_not_found += 1
#     else:
#         num_found += 1
# print(F"Found {num_found} and {num_not_found} not found")

print("Done.")
