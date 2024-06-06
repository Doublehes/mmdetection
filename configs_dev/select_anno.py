#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   debug.py
@Time    :   2024/06/05 11:36:22
@Author  :   shuang.he
@Version :   1.0
@Contact :   shuang.he@momenta.ai
@License :   Copyright 2024, Momenta/shuang.he
@Desc    :   None
'''


import json

anno_path_list = [
    "/home/hs/Documents/mmdetection/data/VOC_coco/voc07_train.json",
    "/home/hs/Documents/mmdetection/data/VOC_coco/voc07_val.json",
    "/home/hs/Documents/mmdetection/data/VOC_coco/voc07_trainval.json",
    "/home/hs/Documents/mmdetection/data/VOC_coco/voc07_test.json",
]

for anno_path in anno_path_list:

    with open(anno_path, 'r') as f:
        anno = json.load(f)
    
    print(f"[Original] image_num: {len(anno['images'])}, annotation_num: {len(anno['annotations'])}")

    # print(anno.keys()) # dict_keys(['images', 'type', 'categories', 'annotations'])
    # images: [{'id': 0, 'file_name': 'VOC2007/JPEGImages/000012.jpg', 'height': 333, 'width': 500}]
    # type: instance
    # categories: [{'supercategory': 'none', 'id': 0, 'name': 'aeroplane'}]
    # annotations: [{'segmentation': [[155, 96, 155, 269, 350, 269, 350, 96]], 'area': 33735, 'ignore': 0, 'iscrowd': 0, 'image_id': 0, 'bbox': [155, 96, 195, 173], 'category_id': 6, 'id': 0}]

    category_id_name = {}
    for item in anno['categories']:
        category_id_name[item['id']] = item['name']

    category_info = {}
    for item in anno['annotations']:
        category_name = category_id_name[item['category_id']]
        if category_name not in category_info:
            category_info[category_name] = {"count": 0}
        category_info[category_name]["count"] += 1

    category_info = sorted(category_info.items(), key=lambda x: x[1]["count"], reverse=True)
    # print(json.dumps(category_info))

    # top_2 = category_info[:2] # [('person', {'count': 2705}), ('car', {'count': 826})]
    # print(top_2)

    select_category = ['person', 'car']
    selected_anno = []
    selected_imgid = set()
    for item in anno['annotations']:
        category_name = category_id_name[item['category_id']]
        if category_name in select_category:
            selected_anno.append(item)
            selected_imgid.add(item['image_id'])

    anno_new = {"images": [], "annotations": [], "categories": [], "type": "instance"}
    category_id_name_new = {
        0: "person",
        1: "car"
    }
    anno_new['categories'] = [{"supercategory": "none", "id": k, "name": v} for k, v in category_id_name_new.items()]

    for item in anno['images']:
        if item['id'] in selected_imgid:
            anno_new["images"].append(item)

    for item in selected_anno:
        item['category_id'] = select_category.index(category_id_name[item['category_id']])
        item['id'] = len(anno_new["annotations"])
        anno_new["annotations"].append(item)


    category_name_str = "_".join(select_category)
    save_path = anno_path.replace(".json", f"_{category_name_str}.json")
    with open(save_path, 'w') as f:
        json.dump(anno_new, f)
    
    print(f"[Selected] image_num: {len(anno_new['images'])}, annotation_num: {len(anno_new['annotations'])}")


