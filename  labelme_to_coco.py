# labelme_to_coco.py

import os
import json
import argparse
import labelme
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as coco_mask

def labelme_to_coco(labelme_dir, output_json, classes):
    """
    Convert Labelme annotations to COCO format.

    Args:
        labelme_dir (str): Directory containing Labelme JSON files.
        output_json (str): Path to save the COCO formatted JSON.
        classes (list): List of class names.
    """
    label_files = [f for f in os.listdir(labelme_dir) if f.endswith('.json')]
    
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 创建类别信息
    for idx, class_name in enumerate(classes, 1):
        coco["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "none"
        })
    
    annotation_id = 1
    image_id = 1
    
    for label_file in tqdm(label_files, desc="Processing Labelme files"):
        label_path = os.path.join(labelme_dir, label_file)
        data = labelme.LabelFile(filename=label_path).load_json()
        
        image_path = os.path.join(labelme_dir, data['imagePath'])
        if not os.path.exists(image_path):
            # 如果图片与 JSON 在同一目录下，尝试拼接路径
            image_path = os.path.join(labelme_dir, data['imagePath'])
            if not os.path.exists(image_path):
                print(f"Image {data['imagePath']} not found. Skipping.")
                continue
        
        img = Image.open(image_path)
        width, height = img.size
        
        coco["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        })
        
        for shape in data['shapes']:
            label = shape['label']
            if label not in classes:
                continue  # 忽略未在类别列表中的标签
            category_id = classes.index(label) + 1
            points = shape['points']
            segmentation = [np.array(points).flatten().tolist()]
            
            # 生成分割掩码
            rle = coco_mask.encode(np.asfortranarray(labelme.utils.shape_to_mask(
                (height, width), points)))
            rle['counts'] = rle['counts'].decode('utf-8')  # 转换为字符串
            
            area = coco_mask.area(rle)
            bbox = coco_mask.toBbox(rle).tolist()  # [x, y, width, height]
            
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0
            })
            annotation_id += 1
        
        image_id += 1
    
    # 保存 COCO 格式 JSON
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=4)
    
    print(f"COCO format JSON saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Labelme JSON annotations to COCO format for YOLOv8.")
    parser.add_argument('--labelme_dir', type=str, required=True, help='Directory containing Labelme JSON files.')
    parser.add_argument('--output_json', type=str, default='coco_annotations.json', help='Output COCO JSON file path.')
    parser.add_argument('--classes', type=str, nargs='+', required=True, help='List of class names.')
    
    args = parser.parse_args()
    
    labelme_to_coco(args.labelme_dir, args.output_json, args.classes)
