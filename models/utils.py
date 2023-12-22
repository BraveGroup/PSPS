import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from types import GeneratorType

import os
import cv2
from panopticapi.utils import IdGenerator
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET

from mmcv.runner.hooks import Hook, HOOKS
from mmdet.core.visualization import get_palette
# training-time visualizer

def expand_target_masks(target_mask, size):
    assert size % 2 == 1, size
    if size <= 1:
        return target_mask
    out_mask = F.max_pool2d(target_mask, size, 1, size//2)
    return out_mask

def partial_cross_entropy_loss(data, target, avg_factor=None):
    num_classes = data.shape[1]

    if target.ndim == 3:
        assert target.dtype == torch.long, (target.shape, target.dtype)
        valid = target < num_classes
        target[~valid] = num_classes
    elif target.ndim == 4:
        valid, target = target.max(1)
        target[valid < 1e-3] = num_classes
    else:
        raise ValueError((target.shape, target.dtype))

    if avg_factor is None:
        avg_factor = valid.sum()
    loss = F.cross_entropy(data, target, ignore_index=num_classes, reduction='none')
    loss = loss.sum() / avg_factor
    return loss

def _unfold_wo_center(x, kernel_size, dilation, with_center=False):
    """
    x: [bsz, c, h, w]
    kernel_size: k
    dilation: d
    return: [bsz, c, k**2-1, h, w]
    """

    assert x.ndim == 4, x.shape
    assert kernel_size % 2 == 1, kernel_size

    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding)

    n, c, h, w = x.shape
    unfolded_x = unfolded_x.reshape(n, c, -1, h, w)

    if with_center:
        return unfolded_x

    # remove the center pixel
    size = kernel_size**2
    unfolded_x = torch.cat((unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), 2)
    return unfolded_x

def _normalized_rgb_to_lab(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    images: [bsz, 3, h, w]
    """
    assert images.ndim == 4, images.shape
    assert images.shape[1] == 3, images.shape

    device = images.device
    mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
    images = (images * std + mean).clip(min=0, max=1)
    rgb = images

    mask = (rgb > .04045).float()
    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)
    xyz_const = torch.as_tensor([
        .412453,.357580,.180423,
        .212671,.715160,.072169,
        .019334,.119193,.950227], device=device).view(3, 3)
    xyz = torch.einsum('mc,bchw->bmhw', xyz_const, rgb)

    sc = torch.as_tensor([0.95047, 1., 1.08883], device=device).view(1, 3, 1, 1)
    xyz_scale = xyz / sc
    mask = (xyz_scale > .008856).float()
    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)
    lab_const = torch.as_tensor([
        0., 116., 0.,
        500., -500., 0.,
        0., 200., -200.], device=device).view(3, 3)
    lab = torch.einsum('mc,bchw->bmhw', lab_const, xyz_int)
    lab[:, 0] -= 16
    return lab.float()

def color_prior_loss(data, images, masks=None, kernel_size=3, dilation=2, avg_factor=None):
    """
    data:   [bsz, classes, h, w] or [bsz, h, w]
    images: [bsz, 3, h, w]
    masks:  [bsz, h, w], (opt.), valid regions
    """
    if data.ndim == 4:
        log_prob = F.log_softmax(data, 1)
    elif data.ndim == 3:
        log_prob = torch.cat([F.logsigmoid(-data[:, None]), F.logsigmoid(data[:, None])], 1)
    else:
        raise ValueError(data.shape)

    B, C, H, W = data.shape
    assert images.shape == (B, 3, H, W), (images.shape, data.shape)
    if masks is not None:
        assert masks.shape == (B, H, W), (masks.shape, data.shape)

    log_prob_unfold = _unfold_wo_center(log_prob, kernel_size, dilation) # [bsz, classes, k**2-1, h, w]
    log_same_prob = log_prob[:, :, None] + log_prob_unfold
    max_ = log_same_prob.max(1, keepdim=True)[0]
    log_same_prob = (log_same_prob - max_).exp().sum(1).log() + max_.squeeze(1) # [bsz, k**2-1, h, w]

    images = _normalized_rgb_to_lab(images)
    images_unfold = _unfold_wo_center(images, kernel_size, dilation)
    images_diff = images[:, :, None] - images_unfold
    images_sim = (-torch.norm(images_diff, dim=1) * 0.5).exp() # [bsz, k**2-1, h, w]

    loss_weight = (images_sim >= 0.3).float()
    if masks is not None:
        loss_weight = loss_weight * masks[:, None]

    #print(data.shape, log_prob.shape, log_same_prob.shape, loss_weight.shape, images_sim.shape)
    loss = -(log_same_prob * loss_weight).sum((1, 2, 3)) / loss_weight.sum((1, 2, 3)).clip(min=1)
    loss = loss.sum() / (len(loss) if avg_factor is None else avg_factor)
    return loss

def neighbour_diff(data, dist=None):
    assert data.ndim == 4
    bsz, c, h, w = data.shape
    neighbour = _unfold_wo_center(data, 3, 1) # [bsz, c, 8, h, w]

    if dist is None:
        return neighbour

    if dist == 'l1':
        diff = (data[:, :, None] - neighbour).abs().sum(1) # [b, 8, h, w]
        return diff
    if dist == 'l2':
        diff = ((data[:, :, None] - neighbour)**2).sum(1) # [b, 8, h, w]
        return diff
    if dist == 'dot':
        diff = 1 - torch.einsum('bchw,bcnhw->bnhw', data, neighbour)
        return diff
    if dist == 'max':
        diff = neighbour.abs().max(1)[0] # [b, 8, h, w]
        return diff
    raise RuntimeError(dist)

def image_to_boundary(images):
    """ images: [bsz, 3, h, w]
    output: [bsz, h, w]
    """
    images = _normalized_rgb_to_lab(images)

    #filter_g = _get_gaussian_kernel(5, 1).to(images).view(1, 1, 5, 5).expand(3, -1, -1, -1)
    #images = F.conv2d(images, filter_g, padding=2, groups=3)

    weight = torch.as_tensor([[
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
        ], [
        1, 2, 1,
        0, 0, 0,
        -1, -2, -1]], dtype=images.dtype, device=images.device)
    weight = weight.view(2, 1, 3, 3).repeat(3, 1, 1, 1)
    edge = F.conv2d(images, weight, padding=1, groups=3)
    edge = (edge**2).mean(1) # [bsz, h, w]
    edge = edge / edge[:, 5:-5, 5:-5].max(2)[0].max(1)[0].view(-1, 1, 1).clip(min=1e-5)
    edge = edge.clip(min=0, max=1)
    #edge = edge / edge.max(1,keepdim=True)[0].max(2, keepdim=True)[0].clip(min=1e-5)
    #mag = mag / mag.max(1, keepdim=True)[0].max(2, keepdim=True)[0].clip(min=1e-5)
    return edge

def cprint(string):
    print(f'\033[95m{string}\033[0m')

def as_list(x):
    return x if isinstance(x, list) else [x]

class VisHelper:
    dataset = 'voc'
    PALETTE_VOID = (225, 225, 196)
    if dataset == 'coco':
        PALETTE = np.array(get_palette('coco', 133)).astype(np.uint8)
    elif dataset == 'voc':
        PALETTE = np.array(get_palette('voc', 20)).astype(np.uint8)
        PALETTE = np.vstack([PALETTE, np.zeros((1, 3), np.uint8)])
    if len(PALETTE < 256):
        _new_palette = np.array([PALETTE_VOID] * 256).astype(np.uint8)
        _new_palette[:len(PALETTE)] = PALETTE
        PALETTE = _new_palette

    @staticmethod
    def visImage(image):
        if isinstance(image, torch.Tensor):
            image = image.data.cpu().numpy()

        assert isinstance(image, np.ndarray), type(image)
        assert image.ndim == 3, image.shape

        # shortcut, if already processed
        if image.dtype == np.uint8:
            return image

        # Guess the dim, if (3, H, W), transpose to (H, W, 3)
        if image.shape[0] == 3 and image.shape[2] != 3:
            image = image.transpose(1, 2, 0)

        img_min = image.min()
        img_max = image.max()
        mean=(123.675, 116.28, 103.53)
        std=(58.395, 57.12, 57.375)

        # Guess: [0, 1] range
        if img_min >= 0 and img_max <= 1:
            image = image * 255

        # Guess: [0, 255]
        if img_min >= 0 and img_max <= 255:
            pass
        else:
            image = image * std + mean

        image = np.clip(image, a_min=0, a_max=255).astype(np.uint8)
        return image

    @staticmethod
    def visPanoptic(result, image=None):
        taken_colors = set()
        def get_color(catId):
            def random_color(base, max_dist=30):
                new_color = base + np.random.randint(low=-max_dist,
                                                     high=max_dist+1,
                                                     size=3)
                return tuple(np.maximum(0, np.minimum(255, new_color)))

            base_color = tuple(VisHelper.PALETTE[catId])
            if base_color not in taken_colors:
                taken_colors.add(base_color)
                return base_color
            else:
                while True:
                    color = random_color(VisHelper.PALETTE[catId])
                    if color not in taken_colors:
                        taken_colors.add(color)
                        return color

        if isinstance(result, torch.Tensor):
            result = result.data.cpu().numpy()
        assert result.ndim == 2, result.shape

        out = np.full(result.shape+(3,), 128, np.uint8)
        for idx in np.unique(result):
            catId = idx % INSTANCE_OFFSET
            mask = result == idx
            color = get_color(catId)
            out[mask] = color

        pan_edge = get_pan_edge(out)

        if image is not None:
            image = VisHelper.visImage(image)
            image = cv2.resize(image, out.shape[:2][::-1])
            out = cv2.addWeighted(image, 0.2, out, 0.8, 0)

        out[pan_edge] = (255, 255, 255)
        return out
    
    @staticmethod
    def visSemantic(result, image=None):
        if isinstance(result, torch.Tensor):
            result = result.data.cpu().numpy()
        assert result.ndim == 2, result.shape

        result_label = result % INSTANCE_OFFSET
        color = VisHelper.PALETTE[result_label.ravel()].reshape(result.shape+(3,))

        if image is not None:
            image = VisHelper.visImage(image)
            image = cv2.resize(image, color.shape[:2][::-1])
            color = cv2.addWeighted(image, 0.2, color, 0.8, 0)
        return color
    
    @staticmethod
    def visBox(image, bboxes):
        def get_color():
            return tuple(random.randint(0, 255) for _ in range(3))

        image = VisHelper.visImage(image)

        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.data.cpu().numpy()
        bboxes = bboxes[:, :4].astype(np.int64)

        out = image.copy()
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            color = get_color()
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 5)

        return out

    @staticmethod
    def vstack(images, width=None, space=3):
        assert isinstance(images, (list, tuple)), type(images)
        assert all([isinstance(x, np.ndarray) for x in images]), [type(x) for x in images]
        images = [x[..., None].repeat(3, -1) if x.ndim == 2 else x for x in images]
        assert all([x.ndim ==3 and x.shape[2] == 3 for x in images]), [x.shape for x in images]

        if len(images) == 0:
            return None
        if len(images) == 1:
            return images[0]

        W = max(x.shape[1] for x in images) if width is None else width
        for i in range(len(images)):
            h, w = images[i].shape[:2]
            H = h * W // w
            images[i] = cv2.resize(images[i], (W, H))

        if space > 0:
            margin = np.full((space, W, 3), 255, np.uint8)
            images = sum([[img, margin] for img in images], [])[:-1]
        out = np.vstack(images)
        return out

    @staticmethod
    def hstack(images, height=None, space=3):
        assert isinstance(images, (list, tuple)), type(images)
        assert all([isinstance(x, np.ndarray) for x in images]), [type(x) for x in images]
        images = [x[..., None].repeat(3, -1) if x.ndim == 2 else x for x in images]
        assert all([x.ndim == 3 and x.shape[2] == 3 for x in images]), [x.shape for x in images]
        out = imvstack([x.transpose(1, 0, 2) for x in images], width=height, space=space)
        if out is not None:
            out = out.transpose(1, 0, 2)
        return out

def _get_rgb_image(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        assert image.ndim == 3 and image.shape[2] == 3, image.shape
        return image

    assert image.ndim == 3 and image.shape[0] == 3, image.shape
    if isinstance(image, torch.Tensor):
        image = image.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image * std + mean
    if image.max() > 1.5:
        image = np.clip(image, a_min=0, a_max=255)
    else:
        image = np.clip(image, a_min=0, a_max=1) * 255
    image = image.astype(np.uint8)
    return image.copy()


def _get_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def visualize_raw(image):
    if isinstance(image, torch.Tensor):
        image = image.data.cpu().numpy()
    assert image.ndim  == 3, image.shape

    if image.shape[0] == 3 and image.shape[2] != 3:
        image = image.transpose(1, 2, 0)
    assert image.shape[2] == 3, image.shape

    if image.dtype in [np.uint8]:
        return image
    if image.max() < 1.001:
        image = (np.clip(image, a_min=0, a_max=1) * 255).astype(np.uint8)
    else:
        image = np.clip(image, a_min=0, a_max=255).astype(np.uint8)
    return image

def visualize_panoptic(image, pan_result, id2rgb, edge=(255, 255, 255)):
    """
    Args:
        image: [tensor | None], if Tensor then of Shape [3, h, w], mean-std removed
        pan_result: tensor, of Shape [h, w]
    Returns:
        pan: np.NDArray, of Shape [h, w, 3], dtype np.uint8
    """
    assert pan_result.ndim == 2, pan_result.shape
    if isinstance(pan_result, torch.Tensor):
        pan_result = pan_result.data.cpu().numpy()
    pan = id2rgb(pan_result).astype(np.uint8) 
    if edge is not None:
        pan_edge = get_pan_edge(pan)

    if image is not None:
        image = _get_rgb_image(image)
        assert image.shape == pan.shape, (image.shape, pan.shape)
        pan = cv2.addWeighted(image, 0.2, pan, 0.8, 0)

    if edge is not None:
        pan[pan_edge] = edge
    return pan

def get_pan_edge(pan):
    assert isinstance(pan, np.ndarray), type(pan)
    assert pan.dtype == np.uint8, pan.dtype
    assert pan.ndim == 3 and pan.shape[-1] == 3, pan.shape

    edges = []
    for c in range(3):
        x = pan[..., c]
        #edge = cv2.Sobel(x, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        edge = cv2.Canny(x, 1, 2)
        edges.append(edge)
    #edges = np.abs(np.array(edges)).max(0) > 0.01
    edges = np.array(edges).max(0)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges > 0

def visualize_bbox(image, bboxes, labels, id2rgb):
    """
    Args:
        image: tensor of Shape [3, h, w]
        bboxes: tensor of shape [N, 4] or [N, 5], each of coordinates 
              (tl_x, tl_y, br_x, br_y) or (tl_x, tl_y, br_x, br_y, score)
        labels: [ tensor | None ], if tensor, then of shape [N,]
    Returns:
        out: np.NDArray, of Shape [h, w, 3], dtype np.uint8
    """

    assert image is not None
    image = _get_rgb_image(image)
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.data.cpu().numpy()
    boxes = bboxes[:, :4].astype(np.int64)
    if labels is not None:
        labels = labels.data.cpu().numpy()

    out = image.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        c = _get_random_color() if labels is None else id2rgb(int(labels[i]))
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 5)

    return out


def visualize_heatmap(image, heatmap, normalize=False):
    assert heatmap.ndim == 2, heatmap.shape
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.data.cpu().numpy()

    if normalize:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)
    heatmap = np.clip(heatmap, a_min=0, a_max=1)
    heatmap = (heatmap * 255.99).astype(np.uint8)
    out = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[..., ::-1]

    if image is not None:
        image = _get_rgb_image(image)
        assert image.shape == out.shape, (image.shape, out.shape)
        out = cv2.addWeighted(image, 0.2, out, 0.8, 0)
    return out

def imvstack(images, width=None, space=3):
    assert isinstance(images, (list, tuple)), type(images)
    assert all([isinstance(x, np.ndarray) for x in images]), [type(x) for x in images]
    images = [x[..., None].repeat(3, -1) if x.ndim == 2 else x for x in images]
    assert all([x.ndim ==3 and x.shape[2] == 3 for x in images]), [x.shape for x in images]

    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]
    
    W = max(x.shape[1] for x in images) if width is None else width
    for i in range(len(images)):
        h, w = images[i].shape[:2]
        H = h * W // w
        images[i] = cv2.resize(images[i], (W, H))
    
    if space > 0:
        margin = np.full((space, W, 3), 255, np.uint8)
        images = sum([[img, margin] for img in images], [])[:-1]
    out = np.vstack(images)
    return out

def imhstack(images, height=None, space=3):
    assert isinstance(images, (list, tuple)), type(images)
    assert all([isinstance(x, np.ndarray) for x in images]), [type(x) for x in images]
    images = [x[..., None].repeat(3, -1) if x.ndim == 2 else x for x in images]
    assert all([x.ndim == 3 and x.shape[2] == 3 for x in images]), [x.shape for x in images]
    out = imvstack([x.transpose(1, 0, 2) for x in images], width=height, space=space)
    if out is not None:
        out = out.transpose(1, 0, 2)
    return out

@HOOKS.register_module()
class SimpleVisHook(Hook):
    def __init__(self, interval):
        self.interval = interval
        self.cache = []

    def should_visualize(self, runner):
        if runner.rank != 0:
            return False
        return self.every_n_iters(runner, self.interval)

    def save_image(self, image, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, image[..., ::-1])

    def after_train_iter(self, runner):
        if not self.should_visualize(runner):
            return
        
        model = runner.model.module
        results = model.bbox_head.get_visualization()
        
        self.cache.append(results)
        self.save_image(results, os.path.join(runner.work_dir,
                                              'visualization',
                                              'last.jpg'))

    def after_train_epoch(self, runner):
        if runner.rank != 0:
            return
        if len(self.cache) == 0:
            return
        
        results = VisHelper.vstack(self.cache)
        self.save_image(results, os.path.join(runner.work_dir,
                                              'visualization',
                                              f'epoch{runner.epoch}.jpg'))
        self.cache = []
        
@HOOKS.register_module()
class VisualizationHook(Hook):
    def __init__(self, dataset, interval):
        self.interval = interval
        self.cache = []

        if dataset is not None:
            self.idgenerator = IdGenerator(dataset.categories)
            self.dataset = dataset
        else:
            raise NotImplementedError

    def _id2color(self, id_map):
        idgenerator = copy.deepcopy(self.idgenerator)
        label2cat = dict((v, k) for k, v in self.dataset.cat2label.items())

        if isinstance(id_map, np.ndarray):
            unique_ids = np.unique(id_map)
            color_lookup = np.zeros((max(unique_ids)+1, 3), np.uint8)
            for i in unique_ids:
                try:
                    L = label2cat[i % INSTANCE_OFFSET]
                except KeyError:
                    # VOID color
                    color_lookup[i] = (225, 225, 196)
                    continue
                color_lookup[i] = idgenerator.get_color(L)
            return color_lookup[id_map.ravel()].reshape(*id_map.shape, 3)
        else:
            return idgenerator.get_color(label2cat[id_map % INSTANCE_OFFSET])

    def _dump_image(self, image, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, image[..., ::-1])

    def _should_visualize(self, runner):
        if runner.rank != 0:
            return False
        return self.every_n_iters(runner, self.interval)

    def _get_visualization(self, result, with_image=True):
        image = _get_rgb_image(result['image'])
        out = [image] if with_image else []
        if 'pan_results' in result:
            for pan_results in as_list(result['pan_results']):
                out.append(visualize_panoptic(image, pan_results, self._id2color))
        if 'bboxes' in result:
            labels = result.get('labels', None)
            if labels is not None:
                labels = as_list(labels)
            for i, bboxes in enumerate(as_list(result['bboxes'])):
                label = None if labels is None else labels[i]
                out.append(visualize_bbox(image, bboxes, label, self._id2color))
        if 'heatmaps' in result:
            for heatmaps in as_list(result['heatmaps']):
                out.append(visualize_heatmap(image, heatmaps))
        if 'raw' in result:
            for raw_image in as_list(result['raw']):
                out.append(visualize_raw(raw_image))

        assert len(out) > 0, result.keys()
        return out

    def after_train_iter(self, runner):
        if not self._should_visualize(runner):
            return

        model = runner.model
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module
        results = model.get_visualization()
        if isinstance(results, dict):
            results = [results]

        out = []
        for i, result in enumerate(results):
            assert isinstance(result, dict), type(result)
            out += self._get_visualization(result, i == 0)
        assert len(out) > 0, [x.keys() for x in results]
        out = imhstack(out, 320)
        self.cache.append(out)
        self._dump_image(out, os.path.join(runner.work_dir, 'visualization', 'last.jpg'))

    def after_train_epoch(self, runner):
        if runner.rank != 0:
            return

        if len(self.cache) == 0:
            return

        out = imvstack(self.cache)
        self._dump_image(out, os.path.join(runner.work_dir, 'visualization', f'epoch{runner.epoch}.jpg'))
        self.cache = []

@HOOKS.register_module()
class VisualizationHook2(Hook):
    def __init__(self, dataset):
        self.interval = 1
        assert dataset is not None
        self.idgenerator = IdGenerator(dataset.categories)
        self.dataset = dataset

    def _id2color(self, id_map):
        idgenerator = copy.deepcopy(self.idgenerator)
        label2cat = dict((v, k) for k, v in self.dataset.cat2label.items())

        if isinstance(id_map, np.ndarray):
            unique_ids = np.unique(id_map)
            color_lookup = np.zeros((max(unique_ids)+1, 3), np.uint8)
            for i in unique_ids:
                try:
                    L = label2cat[i % INSTANCE_OFFSET]
                except KeyError:
                    # VOID color
                    color_lookup[i] = (225, 225, 196)
                    continue
                color_lookup[i] = idgenerator.get_color(L)
            return color_lookup[id_map.ravel()].reshape(*id_map.shape, 3)
        else:
            return idgenerator.get_color(label2cat[id_map % INSTANCE_OFFSET])

    def _dump_image(self, image, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, image[..., ::-1])

    def after_train_iter(self, runner):
        model = runner.model
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module
        results = model.bbox_head.get_result_visualization()

        for res in results:
            name = res['name']
            image = _get_rgb_image(res['image'])

            pan_results = as_list(res['pan_results'])
            pan_results = [visualize_panoptic(image, x, self._id2color) for x in pan_results]

            sem_results = as_list(res['sem_results'])
            sem_results = [visualize_panoptic(image, x, self._id2color, None) for x in sem_results]

            heatmaps = as_list(res['heatmaps'])
            heatmaps = [visualize_heatmap(image, x**5) for x in heatmaps]

            features = visualize_raw(res['feature_proj'])

            demo = [image] + pan_results + sem_results + [features] + heatmaps
            demo = imhstack(demo, space=0)

            self._dump_image(demo, os.path.join(runner.work_dir, 'result_visualization', name+'.jpg'))

    def after_test_iter(self, runner):
        model = runner.model
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module
        results = model.bbox_head.get_test_visualization()

        for res in results:
            name = res['name']
            image = _get_rgb_image(res['image'])
            pan_results = visualize_panoptic(image, res['test_pan_results'], self._id2color)

            demo = imhstack([image, pan_results], 0)
            self._dump_image(demo, os.path.join(runner.work_dir, 'test_visualization', name+'.jpg'))

