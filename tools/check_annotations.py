import json
import os

base_dir = '../COCO_ANIMALS2_lab0.10_val0.08'

out_annotations = os.path.join(base_dir, 'annotations')
out_train2017 = os.path.join(base_dir, 'train2017')
images_in_train2017 = os.listdir(out_train2017)
out_val2017 = os.path.join(base_dir, 'val2017')
images_in_val2017 = os.listdir(out_val2017)
out_unlabeled2017 = os.path.join(base_dir, 'unlabeled2017')
images_in_unlabeled2017 = os.listdir(out_unlabeled2017)

with open(os.path.join(out_annotations, 'instances_train2017.json'), 'r') as f:
    train_annos = json.load(f)

with open(os.path.join(out_annotations, 'instances_val2017.json'), 'r') as f:
    val_annos = json.load(f)

with open(os.path.join(out_annotations, 'instances_unlabeled2017.json'), 'r') as f:
    unlabeled_annos = json.load(f)
    
# Check if all images are in the annotations
print("Checking images in annotations")
# Train
print("Checking train images")
num_found = 0
num_not_found = 0
for image in train_annos['images']:
    json_image = image['file_name']
    if json_image not in images_in_train2017:
        # print(f"Image {json_image} not in train")
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} images and {num_not_found} images not found in train")
# Val
print("Checking val images")
num_found = 0
num_not_found = 0
for image in val_annos['images']:
    json_image = image['file_name']
    if json_image not in images_in_val2017:
        # print(f"Image {json_image} not in val")
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} images and {num_not_found} images not found in val")
# Unlabeled
print("Checking unlabeled images")
num_found = 0
num_not_found = 0
for image in unlabeled_annos['images']:
    json_image = image['file_name']
    if json_image not in images_in_unlabeled2017:
        # print(f"Image {json_image} not in unlabeled")
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} images and {num_not_found} images not found in unlabeled")

# Check if all annotations are in the images
# Train
print("Checking train annotations")
num_found = 0
num_not_found = 0
for image in images_in_train2017:
    json_image = image
    if json_image not in [item['file_name'] for item in train_annos['images']]:
        # print(f"Image {json_image} not in train")
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} images and {num_not_found} images not found in train")
# Val
print("Checking val annotations")
num_found = 0
num_not_found = 0
for image in images_in_val2017:
    json_image = image
    if json_image not in [item['file_name'] for item in val_annos['images']]:
        # print(f"Image {json_image} not in val")
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} images and {num_not_found} images not found in val")

# Unlabeled
print("Checking unlabeled annotations")
num_found = 0
num_not_found = 0
for image in images_in_unlabeled2017:
    json_image = image
    if json_image not in [item['file_name'] for item in unlabeled_annos['images']]:
        # print(f"Image {json_image} not in unlabeled")
        num_not_found += 1
    else:
        num_found += 1
print(F"Found {num_found} images and {num_not_found} images not found in unlabeled")

print("Done.")
