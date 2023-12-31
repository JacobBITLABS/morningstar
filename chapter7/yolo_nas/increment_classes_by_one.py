import json

# Load the COCO annotation file
with open('export.json', 'r') as f:
    coco_data = json.load(f)

# Increment category IDs and annotations
for category in coco_data['categories']:
    category['id'] += 1

for annotation in coco_data['annotations']:
    annotation['category_id'] += 1

# Save the modified COCO annotation file with pretty print
with open('predictions_coco_annotations.json', 'w') as f:
    json.dump(coco_data, f, indent=4)