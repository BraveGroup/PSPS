import torch
import torch.nn.functional as F
from skimage import color

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

@torch.no_grad()
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


def _get_gaussian_kernel(ksize, sigma):
    assert ksize % 2 == 1, ksize
    coords = torch.stack(torch.meshgrid(torch.arange(ksize), torch.arange(ksize)))
    out = torch.exp(-((coords - (ksize // 2))**2).sum(0) / sigma**2)
    return out / out.sum()

@torch.no_grad()
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

#def boundary_loss(data, images, masks, avg_factor=None):
#    """
#    data: [bsz, h, w]
#    images: [bsz, 3, h, w]
#    """
#    B, H, W = data.shape
#    assert images.shape == (B, 3, H, W), (images.shape, data.shape)
#    if masks is not None:
#        assert masks.shape == (B, H, W), (masks.shape, data.shape)
#
#    images = _normalized_rgb_to_lab(images)
#    boundary

