import torch
import numpy as np

def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter, max_visible=None, type='submission'):
  '''
  Returns a frequency mask for position encoding in NeRF.
  
  Args:
    pos_enc_length (int): Length of the position encoding.
    current_iter (int): Current iteration step.
    total_reg_iter (int): Total number of regularization iterations.
    max_visible (float, optional): Maximum visible range of the mask. Default is None. 
      For the demonstration study in the paper.
    
    Correspond to FreeNeRF paper:
      L: pos_enc_length
      t: current_iter
      T: total_iter
  
  Returns:
    torch.array: Computed frequency or visibility mask.
  '''
  if max_visible is None:
    # default FreeNeRF
    if current_iter < total_reg_iter:
      freq_mask = torch.zeros(pos_enc_length)  # all invisible
      ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1 
      ptr = ptr if ptr < pos_enc_length / 3 else pos_enc_length / 3
      int_ptr = int(ptr)
      freq_mask[: int_ptr * 3] = 1.0  # assign the integer part
      freq_mask[int_ptr * 3 : int_ptr * 3 + 3] = (ptr - int_ptr)  # assign the fractional part
      return torch.clip(torch.tensor(freq_mask), 1e-8, 1-1e-8)  # for numerical stability
    else:
      return torch.ones(pos_enc_length)
  else:
    # For the ablation study that controls the maximum visible range of frequency spectrum
    freq_mask = torch.zeros(pos_enc_length)
    freq_mask[: int(pos_enc_length * max_visible)] = 1.0
    return torch.tensor(freq_mask)


def lossfun_occ_reg(rgb, density, reg_range=10, wb_prior=False, wb_range=20):
    '''
    Computes the occulusion regularization loss.

    Args:
        rgb (np.array): The RGB rays/images.
        density (np.array): The current density map estimate.
        reg_range (int): The number of initial intervals to include in the regularization mask.
        wb_prior (bool): If True, a prior based on the assumption of white or black backgrounds is used.
        wb_range (int): The range of RGB values considered to be a white or black background.

    Returns:
        float: The mean occlusion loss within the specified regularization range and white/black background region.
    '''
    # Compute the mean RGB value over the last dimension
    rgb_mean = rgb.mean(-1)
    
    # Compute a mask for the white/black background region if using a prior
    if wb_prior:
        white_mask = np.where(rgb_mean > 0.99, 1, 0) # A naive way to locate white background
        black_mask = np.where(rgb_mean < 0.01, 1, 0) # A naive way to locate black background
        rgb_mask = (white_mask + black_mask) # White or black background
        rgb_mask = rgb_mask.at[:, wb_range:].set(0) # White or black background range
    else:
        rgb_mask = np.zeros_like(rgb_mean)
    
    # Create a mask for the general regularization region
    # It can be implemented as a one-line-code.
    if reg_range > 0:
        rgb_mask = rgb_mask.at[:, :reg_range].set(1) # Penalize the points in reg_range close to the camera
    
    # Compute the density-weighted loss within the regularization and white/black background mask
    return np.mean(density * rgb_mask)


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']
