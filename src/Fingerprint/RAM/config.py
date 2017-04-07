class Config(object):
  win_size = 32
  bandwidth = win_size**2
  batch_size = 32
  eval_batch_size = 50
  loc_std = 0.22

  original_size = 128

  num_channels = 1
  num_classes = 5

  depth = 2
  sensor_size = win_size**2 * depth * num_channels
  minRadius = 8

  # hg_size = hl_size = 128
  # g_size = 256
  # cell_output_size = 256
  # loc_dim = 2
  # cell_size = 256


  hg_size = hl_size = 128
  g_size = 256
  cell_output_size = 256
  loc_dim = 2
  cell_size = 256

  cell_out_size = cell_size
  num_glimpses = 6

  max_grad_norm = 5.

  step = 100000
  lr_start = 1e-3
  lr_min = 1e-4

  # Monte Carlo sampling
  M = 10
