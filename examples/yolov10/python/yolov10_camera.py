import os
import cv2
import sys
import argparse
import torch
from imutils.video import FPS
import time
# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from py_utils.coco_utils import COCO_test_helper
import numpy as np
from threading import Thread

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (640, 640)  # (width, height), such as (1280, 736)

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class WebcamStream :
    def __init__(self, width,height,stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera

        # opening video capture stream
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)

        self.vcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.vcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True

        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing

    # method for starting the thread for grabbing next available frame in input stream
    def start(self):
        self.stopped = False
        self.t.start()

    # method for reading next frame
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method for returning latest read frame
    def read(self):
        return self.frame

    # method called to stop reading frames
    def stop(self):
        self.stopped = True

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def post_process_yolov10(input_data):
    max_det, nc = 300, len(CLASSES)

    boxes, scores = [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        scores.append(input_data[pair_per_branch*i+1])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = torch.from_numpy(np.expand_dims(np.concatenate(boxes), axis=0))
    scores = torch.from_numpy(np.expand_dims(np.concatenate(scores), axis=0))

    max_scores = scores.amax(dim=-1)
    max_scores, index = torch.topk(max_scores, max_det, axis=-1)
    index = index.unsqueeze(-1)
    boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
    scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

    scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
    labels = index % nc
    index = index // nc
    boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

    preds = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

    mask = preds[..., 4] > OBJ_THRESH

    preds = [p[mask[idx]] for idx, p in enumerate(preds)][0]
    boxes = preds[..., :4].numpy()
    scores =  preds[..., 4].numpy()
    classes = preds[..., 5].numpy().astype(np.int64)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def open_cam_usb(dev, width, height):
    vs = cv2.VideoCapture(dev)
    vs.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vs.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    # vs.set(cv2.CAP_PROP_FPS, 10)


    return vs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3566', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')

    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # data params
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    # coco val folder: '../../../datasets/COCO//val2017'
    parser.add_argument('--img_folder', type=str, default='../model', help='img folder path')
    parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)

    # file_list = sorted(os.listdir(args.img_folder))
    # img_list = []
    # for path in file_list:
    #     if img_check(path):
    #         img_list.append(path)

      #Webcam dev /device/video0, /device/video1 etc.
    CAM_DEV = 0

    #Capture Resolution
    CAM_WIDTH = 960
    CAM_HEIGHT = 720

    #Create Stream from Webcam
    # vs = open_cam_usb(CAM_DEV, CAM_WIDTH, CAM_HEIGHT)

    # initializing and starting multi-threaded webcam capture input stream
    webcam_stream = WebcamStream(CAM_WIDTH,CAM_HEIGHT,stream_id=0) #  stream_id = 0 is for primary camera
    webcam_stream.start()

    time.sleep(2.0)
    fps = FPS().start()

    # if not webcam_stream.isOpened():
    #     print("Cannot capture from camera. Exiting.")
    #     quit()

    prev_frame_time = 0
    new_frame_time = 0

    co_helper = COCO_test_helper(enable_letter_box=True)

    while True:
        if webcam_stream.stopped is True :
            break
        else :
            img_src = webcam_stream.read()


        # cv2.imshow("Original", img_src)
        new_frame_time = time.time()
        show_fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        show_fps = int(show_fps)
        show_fps = str("{} FPS".format(show_fps))

        img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        input_data = img
        outputs = model.run([np.expand_dims(input_data, 0)])
        #outputs = model.run([input_data])
        boxes, classes, scores = post_process_yolov10(outputs)



        print('%s' % show_fps)
        if args.img_show:

            img_p = img_src.copy()
            if boxes is not None:
                draw(img_p, co_helper.get_real_box(boxes), scores, classes)

            cv2.putText(img_p, show_fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("full post process result", img_p)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                webcam_stream.stop() # stop the webcam stream
                break



