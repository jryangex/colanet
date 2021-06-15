import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
import os
import copy
import pickle
from json import dumps, loads, JSONEncoder, JSONDecoder
import torch 



class PythonObjectEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
                return JSONEncoder.default(self, obj)
            return {'_python_object':pickle.dumps(obj)}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct




class CocoDetectionEvaluator:
    def __init__(self, dataset):
        assert hasattr(dataset, 'coco_api')
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ['mAP', 'AP_50', 'AP_75', 'AP_small', 'AP_m', 'AP_l']
        
    def xyxy2xywh(bbox):
        """
        change bbox to coco format
        :param bbox: [x1, y1, x2, y2]
        :return: [x, y, w, h]
        """
        return [
                bbox[0].item(),
                bbox[1].item(),
                bbox[2].item() - bbox[0].item(),
                bbox[3].item() - bbox[1].item(),
        ]

    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=self.xyxy2xywh(bbox),
                        score=score)
                    json_results.append(detection)
        return json_results

    def evaluate(self, results, save_dir, rank=-1):
        results_json = self.results2json(results)
        json_path = os.path.join(save_dir, 'results{}.json'.format(rank))
        json.dump(results_json, open(json_path, 'w'))
        coco_dets = self.coco_api.loadRes(json_path)
        coco_eval = COCOeval(copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        aps = coco_eval.stats[:6]
        eval_results = {}
        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
        return eval_results
