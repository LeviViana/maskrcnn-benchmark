import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
from maskrcnn_benchmark.structures.boxlist_ops import *
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import ImageList

class Inferencer(object):
    """
    Implementation of test-time data augmentation.
    """
    def __init__(self, cfg, transforms):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cpu_device = torch.device("cpu")
        self.model.to(self.device)
        self.transforms = transforms
        self.confidence_threshold = 0.7
        self.min_image_size = 800

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.pre_processing = self.build_pre_processing()

    def __call__(self, image):
        return self.inference(image)

    def eval(self):
        self.model.eval()

    def build_pre_processing(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def compute_prediction(self, original_image):
        # apply pre-processing to image
        image = self.pre_processing(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction
    
    def compute_prediction_list(self, image_list):
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

#         predictions = [prediction.resize(size) for prediction, size in zip(predictions, image_list.image_sizes)]
        
        return predictions

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """

        image_ = image.copy()
        predictions = self.select_top_predictions(predictions)
        # labels = predictions.get_field("labels")
        predictions = predictions.resize(predictions.size)
        boxes = predictions.bbox

        for box in boxes:
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image_ = cv2.rectangle(
                image_, tuple(top_left), tuple(bottom_right), (0, 0, 255), 1
            )

        return image_

    def inference(self, image):
        if isinstance(image, ImageList):
            return self._inference_list(image)
        else:
            return self._inference(image)
    
    def _inference_list(self, image_list):
        predictions = self.compute_prediction_list(image_list)
        
        tensors_flip = torch.empty((0, *image_list.tensors.size()[1:]))
        for t in image_list.tensors:
            tmp = T.ToTensor()(T.functional.hflip(T.ToPILImage()(t)))
            tmp = self.pre_processing(tmp)
            tensors_flip = torch.cat((tensors_flip, tmp.unsqueeze(0)))
        
        image_list_flip = image_list
        image_list_flip.tensors = tensors_flip
        
        predictions_flip = self.compute_prediction_list(image_list_flip)

        pred1 = [p.transpose(0) for p in predictions_flip]
        pred2 = predictions
        
        result = []
        
        for p1, p2 in zip(pred1, pred2):
            result.append(self.merge_predictions(p1, p2))

        return result
    
    def _inference(self, image):
        predictions = self.compute_prediction(image)
        image_flip = image[:, ::-1, :]
        predictions_flip = self.compute_prediction(image_flip)

        pred1 = predictions_flip.transpose(0)
        pred2 = predictions

        return self.merge_predictions(pred1, pred2)

    def merge_predictions(self, pred1, pred2):
        # for one Image at a time
        img_size = pred1.size
        
        scores1 = pred1.get_field("scores")
        scores2 = pred2.get_field("scores")
        
        labels1 = pred1.get_field("labels")
        labels2 = pred2.get_field("labels")

        bl1 = BoxList(pred1.bbox, img_size)
        bl1.add_field("scores", scores1)
        bl2 = BoxList(pred2.bbox, img_size)
        bl2.add_field("scores", scores2)

        iou_boxes = (boxlist_iou(bl1, bl2) > 0.7)
        if iou_boxes.numel() > 0:
            crossed_boxes = (pred1.bbox[:, None, :] + pred2.bbox) / 2
            crossed_scores = torch.max(scores1[:, None], scores2)
            crossed_labels = torch.max(labels1[:, None], labels2)

            boxes1 = pred1.bbox[1 - iou_boxes.max(1)[0]]
            boxes2 = pred2.bbox[1 - iou_boxes.max(0)[0]]
            box_merged = crossed_boxes[iou_boxes]

            scores1 = scores1[1 - iou_boxes.max(1)[0]]
            scores2 = scores2[1 - iou_boxes.max(0)[0]]
            scores_merged = crossed_scores[iou_boxes]
            
            labels1 = labels1[1 - iou_boxes.max(1)[0]]
            labels2 = labels2[1 - iou_boxes.max(0)[0]]
            labels_merged = crossed_labels[iou_boxes]
            
            all_boxes = torch.cat((boxes1, boxes2, box_merged), dim=0)
            all_scores = torch.cat((scores1, scores2, scores_merged), dim=0)
            all_labels = torch.cat((labels1, labels2, labels_merged), dim=0)

            final_boxlist = BoxList(all_boxes, img_size)
            final_boxlist.add_field("scores", all_scores)
            final_boxlist.add_field("labels", all_labels)
            final_boxlist = boxlist_nms(final_boxlist, 0.7, score_field="scores")
            
        else:
            final_boxlist = BoxList(torch.empty((0, 4)), img_size)
            final_boxlist.add_field("scores", torch.empty((0, 1)))
            final_boxlist.add_field("labels", torch.empty((0, 1)))

        return final_boxlist
    