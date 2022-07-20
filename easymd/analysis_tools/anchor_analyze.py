from mmdet.core import build_anchor_generator
import mmdet
import mmcv
import numpy as np
import time
import cv2 as cv

def show_anchor(input_shape_hw, stride, anchor_generator_cfg, random_n, select_n):
    img = np.zeros(input_shape_hw, np.uint8)
    feature_map = []
    for s in stride:
        feature_map.append([input_shape_hw[0] // s, input_shape_hw[1] // s])
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    anchors = anchor_generator.grid_anchors(feature_map)  # 输出原图尺度上anchor坐标 xyxy格式 左上角格式
    base_anchors = anchor_generator.base_anchors
    
    for i,each in enumerate(base_anchors):
        each[:,0:4:2] += input_shape_hw[0]//2
        each[:,1:4:2] += input_shape_hw[1]//2
    for _ in range(random_n):
        disp_img = []
        for i,anchor in enumerate(anchors):
            img = np.zeros(input_shape_hw, np.uint8)
            anchor = anchor.cpu().numpy()
            print(anchor.shape)
            index = (anchor[:, 0] > 0) & (anchor[:, 1] > 0) & (anchor[:, 2] < input_shape_hw[1]) & \
                    (anchor[:, 3] < input_shape_hw[0])
            anchor = anchor[index]
            
            anchor = np.random.permutation(anchor)
            img_ = mmcv.imshow_bboxes(img, anchor[:select_n], thickness=1, show=False)
            img_ = mmcv.imshow_bboxes(img_, base_anchors[i].cpu().numpy(), thickness=1, colors='red', show=False)
            #disp_img.append(img_)
            cv.imshow('img',img_)
            if cv.waitKey(0) & 0xFF== ord('q'):
                exit(0)
            #time.sleep(0.3)

def demo_retinanet(input_shape_hw):
    stride = [8, 16, 32, 64, 128]
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        octave_base_scale=4,  # 每层特征图的base anchor scale,如果变大，则整体anchor都会放大
        scales_per_octave=3,  # 每层有3个尺度 2**0 2**(1/3) 2**(2/3)
        ratios=[0.5, 1.0, 2.0],  # 每层的anchor有3种长宽比 故每一层每个位置有9个anchor
        strides=stride)  # 每个特征图层输出stride,故anchor范围是4x8=32,4x128x2**(2/3)=812.7
    random_n = 10
    select_n = 100
    show_anchor(input_shape_hw, stride, anchor_generator_cfg, random_n, select_n)


def demo_yolov3(input_shape_hw):
    stride = [32, 16, 8]
    anchor_generator_cfg = dict(
        type='YOLOAnchorGenerator',
        base_sizes=[[(116, 90), (156, 198), (373, 326)],
                    [(30, 61), (62, 45), (59, 119)],
                    [(10, 13), (16, 30), (33, 23)]],
        strides=stride)

    random_n = 10
    select_n = 100
    show_anchor(input_shape_hw, stride, anchor_generator_cfg, random_n, select_n)


if __name__ == '__main__':
    input_shape_hw = (320, 320, 3)
    demo_retinanet(input_shape_hw)
    #demo_yolov3(input_shape_hw)