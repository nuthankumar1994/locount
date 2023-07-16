
import os
import json
from PIL import Image
from tqdm import tqdm
import pandas as pd

def txt_to_csv(path):
    # Set the path to the directory containing the text files
    txt_files_directory = path
    # Create an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Iterate over the text files in the directory
    for file_name in os.listdir(txt_files_directory):
        if file_name.endswith('.txt'):
            file_path = os.path.join(txt_files_directory, file_name)
            
            # Load the text file into a DataFrame
            df = pd.read_csv(file_path, delimiter=',', header=None, names=['x1','y1', 'x2','y2', 'class', 'cnt'])
            
            # Add a column for the file name
            df['File Name'] = file_name
            
            # Append the data to the combined DataFrame
            combined_data = combined_data.append(df, ignore_index=True)

    # Set the path to the output CSV file
    output_csv_file = 'annotation_train.csv'
    # Save the combined data to the CSV file
    combined_data.to_csv(output_csv_file, index=False)
    return 'converstion done'

def yolo_to_coco(image_dir, label_dir, output_dir):
	# Define categories

	# Initialize data dict
	data = {'train': [], 'validation': [], 'test': []}

	# Loop over splits
	for split in ['train']:
		split_data = {'info': {}, 'licenses': [], 'images': [], 'annotations': [], 'categories': []}

		# Get image and label files for current split
		image_files = sorted(os.listdir(image_dir))
		label_files = sorted(os.listdir(label_dir))
		# print(image_files)

		# Loop over images in current split
		cumulative_id = 0
		with tqdm(total=len(image_files), desc=f'Processing {split} images') as pbar:
			for i, filename in enumerate(image_files):
				image_path = os.path.join(image_dir, filename)
				im = Image.open(image_path)
				im_id = i + 1

				split_data['images'].append({
					'id': im_id,
					'file_name': filename,
					'width': im.size[0],
					'height': im.size[1]
				})
				
				# Get labels for current image
				label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')
				with open(label_path, 'r') as f:
					yolo_data = f.readlines()
				

				for line in yolo_data:
					x1, y1, x2, y2, class_name, cnt = line.split(sep=',')
					# bbox_x = x1
					# bbox_y = y1
					# width = float(x2) - float(x1)
					# height = float(y2) - float(y1)
					count = cnt
					# bbox_width = float(width) * im.size[0]
					# bbox_height = float(height) * im.size[1]
					
					category_id = next((cat['id'] for cat in split_data['categories'] if cat['name'] == class_name), None)
					if category_id is None:
						category_id = len(split_data['categories']) + 1
						split_data['categories'].append({
							'id': category_id,
							'name': class_name,
							'supercategory': 'object'
					})

					x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
					split_data['annotations'].append( {
					'id': len(split_data['annotations']) + 1,
					'image_id': im_id,
					'category_id': category_id,
					'bbox': [x1, y1, x2 - x1, y2 - y1],
					'area': (x2 - x1) * (y2 - y1),
					'iscrowd': 0,
					'gt_count': count
				})
					cumulative_id += 1

				pbar.update(1)

		data[split] = split_data

	# Save data to JSON files
	filename = os.path.join(output_dir, f'annotation_train.json')
	with open(filename, 'w') as f:
		json.dump({data[split]}, f)
	return data

image_dir = 'coco/Locount_ImagesTrain'
label_dir = 'coco/Locount_GtTxtsTrain'
output_dir = './'
coco_data = yolo_to_coco(image_dir, label_dir, output_dir)




import json

def write_dict_to_json(data, json_path):
    with open(json_path, 'w') as file:
        json.dump(data, file)

def read_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

# Example usage
json_path = 'coco/annotations/annotation_test.json'  # Specify the path and filename of your JSON file
json_data = read_json(json_path)
json_data = json_data['data']

write_dict_to_json(json_data,json_path)

# Access the data
# Assuming the JSON file has the same structure as the COCO format
# annotations = json_data['annotations']
# images = json_data['images']
categories = json_data['categories']
print(categories)
# print(json_data)
# if json_data == None:
#     print()
# Process the data as needed
# ...