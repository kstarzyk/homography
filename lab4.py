import numpy as np
from scipy import linalg
import math
from utils import *

def bi_interpolate(I, y, x):
  x0,x1 = int(math.floor(x)), int(math.ceil(x))
  y0,y1 = int(math.floor(y)), int(math.ceil(y))

  if x0 == x1:
    a = pix(I, y0, x0)
    b = pix(I, y1, x0)
  else:
    a = (x1-x) * pix(I, y0, x0) + (x-x0) * pix(I, y0, x1)
    b = (x1-x) * pix(I, y1, x0) + (x-x0) * pix(I, y1, x1)

  return a if y0 == y1 else (y1 - y) * a + (y - y0) * b

def apply_homography(src, out, H, bilinear=False):
  H_inv = linalg.inv(H)
  for y in range(len(out)):
    for x in range(len(out[0])):
      yp, xp, w = H_inv.dot(np.array([y, x, 1.]))
      yp, xp = (yp/w, xp/w)
      if yp >= 0 and yp < src.shape[0] and xp >= 0  and xp < src.shape[1]:
        # :3 neet do be set because of RGBa possibility (ignore)
        if bilinear:
          out[y,x,:3] = bi_interpolate(src, yp, xp)[:3]
        else:
          out[y,x,:3] = src[int(round(yp)), int(round(xp))][:3]
  return out

def apply_homography_robust(src, out, H, bilinear=False):
  # same as apply_homography but with different it range
  box = compute_box(src.shape, H)
  a1,b1 = box[0][0], box[1][0]
  a2,b2 = box[0][1], box[1][1]
  Hinv = linalg.inv(H)
  for y in xrange(a1,b1):
    for x in xrange(a2,b2):
      yp, xp, w = Hinv.dot(np.array([y, x, 1.0]))
      yp, xp = (yp / w, xp / w)
      if yp >= 0 and yp < src.shape[0] and xp >= 0  and xp < src.shape[1]:
        # :3 neet do be set because of RGBa possibility (ignore)
        if bilinear:
          out[y,x,:3] = bi_interpolate(src, yp, xp)[:3]
        else:
          out[y,x,:3] = src[int(round(yp)), int(round(xp))][:3]
  return out

def compute_homography(pairs):
  A = np.zeros((9,9))
  for i, pair in enumerate(pairs):
    add_equations(A, i, pair)
  # dirty but works
  A[8,8] = 1.
  x = np.array([0.]*8 + [1.])
  return linalg.inv(A).dot(x).reshape([3,3])

def stitch(Ia, Ib, pairs):
  # glues im1 and im2, returns output (can be bigger than passed imgs)
  # assumes Bilineal=True
  H = compute_homography(pairs)
  box = union_box(compute_box(Ia.shape, H), [[0,0], [Ib.shape[0], Ib.shape[1]]])
  tr = translate(box)
  tr_inv = linalg.inv(tr)
  out = np.zeros((box[1][0]-box[0][0],box[1][1]-box[0][1], 3))
  w, h = len(out[0]), len(out)
  for y in range(h):
    for x in range(w):
      yt, xt, wt = tr_inv.dot(np.array([y, x, 1.]))
      if yt >= 0 and yt < Ib.shape[0] and xt >= 0 and xt < Ib.shape[1]:
        out[y,x] = bi_interpolate(Ib, yt, xt)
  apply_homography_robust(Ia, out, tr.dot(H) , True)
  return out


