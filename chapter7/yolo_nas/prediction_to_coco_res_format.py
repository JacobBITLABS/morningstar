import json

# Load the COCO annotation file
with open('predictions_coco_annotations.json', 'r') as f:
    coco_data = json.load(f)

# Extract annotations
annotations = coco_data['annotations']

# Create a new dictionary with the list of annotations
new_data = {'annotations_list': annotations}

# Save the annotations list to a new JSON file
with open('predictions_coco_annotations_res_format.json', 'w') as f:
    json.dump(new_data, f, indent=4)