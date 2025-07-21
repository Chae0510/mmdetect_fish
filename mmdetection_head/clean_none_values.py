#!/usr/bin/env python3
"""
Script to clean JSON files by removing images with None values for VBN, PH, or clf_score.
"""

import json
import os

def clean_json_file(json_path):
    """Remove images with None values from JSON file."""
    print(f"Processing {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    original_count = len(data['images'])
    removed_count = 0
    removed_images = []
    
    # Filter out images with None values
    cleaned_images = []
    for img in data['images']:
        if (img.get('VBN') is None or 
            img.get('PH') is None or 
            img.get('clf_score') is None):
            removed_count += 1
            removed_images.append({
                'id': img.get('id'),
                'file_name': img.get('file_name'),
                'VBN': img.get('VBN'),
                'PH': img.get('PH'),
                'clf_score': img.get('clf_score')
            })
        else:
            cleaned_images.append(img)
    
    data['images'] = cleaned_images
    
    # Update annotations to only include those from remaining images
    remaining_img_ids = set(img['id'] for img in cleaned_images)
    cleaned_annotations = []
    for ann in data['annotations']:
        if ann['image_id'] in remaining_img_ids:
            cleaned_annotations.append(ann)
    
    data['annotations'] = cleaned_annotations
    
    # Save cleaned data
    backup_path = json_path.replace('.json', '_backup.json')
    os.rename(json_path, backup_path)
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Original images: {original_count}")
    print(f"Removed images: {removed_count}")
    print(f"Remaining images: {len(cleaned_images)}")
    print(f"Remaining annotations: {len(cleaned_annotations)}")
    print(f"Backup saved to: {backup_path}")
    
    if removed_images:
        print("\nRemoved images:")
        for img in removed_images:
            print(f"  ID {img['id']}: {img['file_name']} (VBN={img['VBN']}, PH={img['PH']}, clf_score={img['clf_score']})")
    
    return removed_count

def main():
    """Clean all JSON files in the subset_uniform directory."""
    base_dir = "/workspace/fish_data/subset_uniform"
    
    json_files = [
        "vbn_ph_uniform_train.json",
        "vbn_ph_uniform_val.json", 
        "vbn_ph_uniform_test.json"
    ]
    
    total_removed = 0
    
    for json_file in json_files:
        json_path = os.path.join(base_dir, json_file)
        if os.path.exists(json_path):
            removed = clean_json_file(json_path)
            total_removed += removed
            print("-" * 50)
        else:
            print(f"File not found: {json_path}")
    
    print(f"\nTotal images removed: {total_removed}")

if __name__ == "__main__":
    main() 