#!/usr/bin/env python3
"""
Script to find images with None values for VBN, PH, or clf_score and save them to a text file.
"""

import json
import os
from datetime import datetime

def find_none_values(json_path):
    """Find images with None values in JSON file."""
    print(f"Processing {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    none_images = []
    
    for img in data['images']:
        if (img.get('VBN') is None or 
            img.get('PH') is None or 
            img.get('clf_score') is None):
            none_images.append({
                'id': img.get('id'),
                'file_name': img.get('file_name'),
                'VBN': img.get('VBN'),
                'PH': img.get('PH'),
                'clf_score': img.get('clf_score')
            })
    
    return none_images

def main():
    """Find None values in all JSON files and save to text file."""
    base_dir = "/workspace/fish_data/subset_uniform"
    
    json_files = [
        "vbn_ph_uniform_train.json",
        "vbn_ph_uniform_val.json", 
        "vbn_ph_uniform_test.json"
    ]
    
    all_none_images = []
    
    for json_file in json_files:
        json_path = os.path.join(base_dir, json_file)
        if os.path.exists(json_path):
            none_images = find_none_values(json_path)
            for img in none_images:
                img['source_file'] = json_file
            all_none_images.extend(none_images)
            print(f"Found {len(none_images)} images with None values in {json_file}")
        else:
            print(f"File not found: {json_path}")
    
    # Save to text file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(base_dir, f"none_values_{timestamp}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Images with None values found on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        if all_none_images:
            f.write(f"Total images with None values: {len(all_none_images)}\n\n")
            
            for img in all_none_images:
                f.write(f"Source: {img['source_file']}\n")
                f.write(f"Image ID: {img['id']}\n")
                f.write(f"File Name: {img['file_name']}\n")
                f.write(f"VBN: {img['VBN']}\n")
                f.write(f"PH: {img['PH']}\n")
                f.write(f"clf_score: {img['clf_score']}\n")
                f.write("-" * 40 + "\n")
        else:
            f.write("No images with None values found.\n")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total images with None values: {len(all_none_images)}")

if __name__ == "__main__":
    main() 