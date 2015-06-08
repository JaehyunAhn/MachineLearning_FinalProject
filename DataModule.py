"""
    Machine Learning Final Project
    - Author : Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
    - Due Date : 15/06/16
"""

import numpy as np

def collect_data(object_dataset, label_name, target):
    for items in object_dataset:
        array = np.asarray(items)
        target['data'].append(array)
        target['label'].append(label_name)
    return target