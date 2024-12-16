import numpy as np
import torch


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fft2(data):
    # assert data.size(-1) == 2
    # data = torch.fft(data, 2, normalized=False)
    # data = fftshift(data, dim=(-3, -2))
    data = torch.fft.fftshift(torch.fft.fft2(data))
    return data


def ifft2(data):
    # assert data.size(-1) == 2
    # data = ifftshift(data, dim=(-3, -2))
    # data = torch.ifft(data, 2, normalized=False)
    data = torch.fft.ifft2(torch.fft.ifftshift(data))
    return data


def gaussian_mask(data, R):
    nB = data.size(0)
    nY = data.size(2)
    nX = data.size(3)
    mask = np.zeros((nB, 1, nY, nX), dtype=np.float32)

    if R > 1:
        # nACS = max(round(nY / (R ** 2)), round(nY * 0.1))
        nACS = round(nY * 0.1)
        ACS_s = round((nY - nACS) / 2)
        ACS_e = ACS_s + nACS
        mask[:, :, ACS_s:ACS_e, :] = 1

        nSamples = int(nY / R)
        for b in range(nB):
            r = np.floor(np.random.normal(nY / 2, nY * (15.0 / 128), nSamples))
            r = np.clip(r.astype(int), 0, nY - 1)
            mask[b, :, r.tolist(), :] = 1

    mask = torch.from_numpy(mask).to(data.device)
    # mask = mask.unsqueeze(-1)
    # mask = torch.cat([mask, mask], dim=-1)
    return mask


def uniform_random_mask(data, R):
    nB = data.size(0)
    nY = data.size(2)
    mask = np.zeros(data.size(), dtype=np.float32)

    if R > 1:
        nACS = round(nY / (R ** 2))
        ACS_s = round((nY - nACS) / 2)
        ACS_e = ACS_s + nACS
        mask[:, :, ACS_s:ACS_e, :] = 1

        nSamples = int(nY / R)
        for b in range(nB):
            r = np.random.randint(nY, size=nSamples)
            mask[b, :, r.tolist(), :] = 1

    mask = torch.from_numpy(mask).to(data.device)
    # mask = mask.unsqueeze(-1)
    # mask = torch.cat([mask, mask], dim=-1)
    return mask


def uniform_mask(data, R, idx=0):
    nY = data.size(2)
    mask = np.zeros(data.size(), dtype=np.float32)
    idx = idx % R

    nACS = round(nY / (R ** 2))
    ACS_s = round((nY - nACS) / 2)
    ACS_e = ACS_s + nACS
    mask[:, :, ACS_s:ACS_e, :] = 1
    mask[:, :, idx::R, :] = 1

    mask = torch.from_numpy(mask).to(data.device)
    # mask = mask.unsqueeze(-1)
    # mask = torch.cat([mask, mask], dim=-1)
    return mask


def center_mask(data):
    nB = data.size(0)
    nY = data.size(2)
    nX = data.size(3)
    mask = np.zeros((nB, 1, nY, nX), dtype=np.float32)

    # nACS = max(round(nY * R), round(nY * 0.1))
    nACS = round(nY * 0.1)
    ACS_s = round((nY - nACS) / 2)
    ACS_e = ACS_s + nACS
    mask[:, :, ACS_s:ACS_e, :] = 1

    # mask = np.zeros(nY, dtype=np.float32)
    # nACS = round(nY * R)
    # ACS_s = round((nY - nACS) / 2)
    # ACS_e = ACS_s + nACS
    # mask[:ACS_s] = 1 / ACS_s * np.linspace(0, ACS_s, num=ACS_s, endpoint=False)
    # mask[ACS_s:ACS_e] = 1
    # mask[ACS_e:] = -1 / (nY - ACS_e) * np.linspace(ACS_e, nY, num=nY - ACS_e, endpoint=False) + nY / (nY - ACS_e)
    # mask = np.tile(mask[np.newaxis, np.newaxis, :, np.newaxis], (data.size(0), data.size(1), 1, data.size(3)))

    mask = torch.from_numpy(mask).to(data.device)
    # mask = mask.unsqueeze(-1)
    # mask = torch.cat([mask, mask], dim=-1)
    return mask


def step_mask(data, nstep):
    nY = data.size(2)
    mask = np.zeros([nstep, data.size(1), data.size(2), data.size(3)], dtype=np.float32)

    nlines = nY // nstep
    for n in range(nstep):
        mask[n, :, n * nlines:min((n + 1) * nlines, nY), :] = 1.0

    mask = torch.from_numpy(mask).to(data.device)
    return mask


def mask_fn(type='gaussian'):
    if type == 'gaussian':
        return gaussian_mask
    elif type == 'uniform_random':
        return uniform_random_mask
    elif type == 'center':
        return center_mask
    else:
        raise NotImplementedError('Mask pattern [%s] is not implemented' % type)
