import torch as ch
import numpy as np
import torch.nn as nn
import itertools
from kornia.geometry.transform import resize, rotate, center_crop, translate
from cleverhans.future.torch.utils import get_or_guess_labels


def _apply_black_border(x, border_size):
  orig_height = x.shape[2]
  orig_width  = x.shape[3]
  x = resize(x, (orig_width - 2*border_size,
                                 orig_height - 2*border_size))
  return nn.ConstantPad2d(border_size, 0)(x)


def apply_transformation(x, trans):
  dx, dy, angle = trans[0], trans[1], trans[2]
  height, width = x.shape[2], x.shape[3]

  # Pad the image to prevent two-step rotation / translation from truncating
  # corners
  max_dist_from_center = np.sqrt(height**2+width**2) / 2
  min_edge_from_center = float(np.min([height, width])) / 2
  padding = np.ceil(max_dist_from_center -
                    min_edge_from_center).astype(np.int32)
  x = nn.ConstantPad2d(padding, 0)(x)

  # Apply rotation
  angle = ch.from_numpy(np.ones(x.shape[0]) * angle )
  angle = angle.to(x.get_device())
  x = rotate(x, angle)

  # Apply translation
  dx_in_px = -dx * height
  dy_in_px = -dy * width
  translation = ch.from_numpy(np.tile(np.array([dx_in_px, dy_in_px], dtype=np.float32), (x.shape[0], 1)))
  translation = translation.to(x.get_device())
  x = translate(x, translation)
  x = translate(x, translation)
  # Pad if needed
  if x.shape[2] < height or x.shape[3] < width:
    pad = nn.ConstantPad2d((0, max(0, height - x.shape[2]), 0, max(0, width - x.shape[3])), 0)
    x = pad(x)
  return center_crop(x, (height, width))


def spatial_transformation_method(model, x, n_samples=None,
                                  dx_min=-0.1, dx_max=0.1, n_dxs=2,
                                  dy_min=-0.1, dy_max=0.1, n_dys=2,
                                  angle_min=-30, angle_max=30, n_angles=6,
                                  black_border_size=0, **kwargs):
  """
  :param n_samples: (optional) The number of transformations sampled to
                    construct the attack. Set it to None to run
                    full grid attack.
  :param dx_min: (optional float) Minimum translation ratio along x-axis.
  :param dx_max: (optional float) Maximum translation ratio along x-axis.
  :param n_dxs: (optional int) Number of discretized translation ratios
                along x-axis.
  :param dy_min: (optional float) Minimum translation ratio along y-axis.
  :param dy_max: (optional float) Maximum translation ratio along y-axis.
  :param n_dys: (optional int) Number of discretized translation ratios
                along y-axis.
  :param angle_min: (optional float) Largest counter-clockwise rotation
                    angle.
  :param angle_max: (optional float) Largest clockwise rotation angle.
  :param n_angles: (optional int) Number of discretized angles.
  :param black_border_size: (optional int) size of the black border in pixels.
  """

  if dx_min < -1 or dy_min < -1 or dx_max > 1 or dy_max > 1:
      raise ValueError("The value of translation must be bounded "
                       "within [-1, 1]")
  
  y = get_or_guess_labels(model, x, **kwargs)

  # Define the range of transformations
  dxs = np.linspace(dx_min, dx_max, n_dxs)
  dys = np.linspace(dy_min, dy_max, n_dys)
  angles = np.linspace(angle_min, angle_max, n_angles)

  if n_samples is None:
    transforms = list(itertools.product(*[dxs, dys, angles]))
  else:
    sampled_dxs = np.random.choice(dxs, n_samples)
    sampled_dys = np.random.choice(dys, n_samples)
    sampled_angles = np.random.choice(angles, n_samples)
    transforms = zip(sampled_dxs, sampled_dys, sampled_angles)

  x = _apply_black_border(x, black_border_size)
  transformed_ims = []
  for transform in transforms:
    transformed_ims.append(apply_transformation(x, transform))
  transformed_ims = ch.stack(transformed_ims)

  def _compute_xent(x):
    logits = model(x)
    loss = nn.CrossEntropyLoss(reduction='none')
    return loss(logits, y)

  all_xents = ch.stack([_compute_xent(x) for x in transformed_ims])
  
  # We want the worst case sample, with the largest xent_loss
  worst_sample_idx = ch.argmax(all_xents, 0)  # B
  transformed_ims_perm = ch.transpose(transformed_ims, 0, 1)
  after_lookup = []
  for i, sample_id in enumerate(worst_sample_idx):
    after_lookup.append(transformed_ims_perm[i][sample_id])
  after_lookup = ch.stack(after_lookup)

  return after_lookup
