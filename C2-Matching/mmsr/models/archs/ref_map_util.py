import torch.nn.functional as F


def sample_patches(inputs, patch_size=3, stride=1):
    """Extract sliding local patches from an input feature tensor.
    The sampled pathes are row-major.

    Args:
        inputs (Tensor): the input feature maps, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.

    Returns:
        patches (Tensor): extracted patches, shape:
        (c, patch_size, patch_size, n_patches).
    """

    """
    unfold(dim, size, step) 返回的是tensor的维度dim的切片，切片大小size, 步长step  
    [c, (h - patch_size) / step + 1, w, patch_size]
    [c, (h - patch_size) / step + 1, (w - patch_size) / step + 1, patch_size, patch_size])]
    [c, n_patches, patch_size, patch_size]  , where n_patches is equal to [(h - patch_size) / step + 1] * [(w - patch_size) / step + 1]
    [c, patch_size, patch_size, n_patches]
    """

    c, h, w = inputs.shape
    patches = inputs.unfold(1, patch_size, stride)\
                    .unfold(2, patch_size, stride)\
                    .reshape(c, -1, patch_size, patch_size)\
                    .permute(0, 2, 3, 1)
    return patches

def feature_match_index(feat_input,
                        feat_ref,
                        patch_size=3,
                        input_stride=1,
                        ref_stride=1,
                        is_norm=True,
                        norm_input=False):
    """Patch matching between input and reference features.

    Args:
        feat_input (Tensor): the feature of input, shape: (c, h, w).
        feat_ref (Tensor): the feature of reference, shape: (c, h, w).   这里显式的假设了输入图像和参考图像的特征图的大小相同
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
        is_norm (bool): determine to normalize the ref feature or not.
            Default:True.

    Returns:
        max_idx (Tensor): The indices of the most similar patches.
        max_val (Tensor): The correlation values of the most similar patches.
    """

    # patch decomposition, shape: (c, patch_size, patch_size, n_patches)
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)

    # normalize reference feature for each patch in both channel and
    # spatial dimensions.

    # batch-wise matching because of memory limitation
    _, h, w = feat_input.shape
    batch_size = int(1024.**2 * 512 / (h * w))
    n_patches = patches_ref.shape[-1]

    max_idx, max_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size] # [c, patch_size, patch_size, batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5) # 因为是通过卷积计算相似度的，卷积得到的结果可能受到特征向量的本来的长度的影响，因此这里根据is_norem选择进行归一化处理
        corr = F.conv2d(               # F.conv2d(C_out, C_in, K_size, K_size) 其中C_out为输出通道数，即卷积核数量，也就是patches数量；C_in为输入通道数，即输入图像的特征向量的长度
            feat_input.unsqueeze(0),   # [1, c, h, w]
            batch.permute(3, 0, 1, 2), # [batch_size, c, patch_size, patch_size] 类似卷积核, 每个特征块和输入特征做卷积都能计算出一个矩阵的相似度
            stride=input_stride)       # output:[1, batch_size(or say C_out), h-2, w-2] 注意这里没有pad

        max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0) # [1, h-2, w-2] idx[0~batch_size-1]

        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp
        else:
            indices = max_val_tmp > max_val
            max_val[indices] = max_val_tmp[indices]
            max_idx[indices] = max_idx_tmp[indices] + idx # 这里面的idx显然是参考图像特征经过sample后得到的一个序列patches，然后对应序列的一个索引

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(
            int((h - patch_size) / input_stride + 1),
            int((w - patch_size) / input_stride + 1))
        max_val = max_val / norm

    return max_idx, max_val
