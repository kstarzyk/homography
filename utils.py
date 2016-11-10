import numpy as np
from PIL import Image

DEST = './Result/'

def read_img(name):
  return np.asarray(Image.open(name), dtype=np.float32)

def save_img(arr, name):
  Image.fromarray(arr.round().astype(np.uint8)).save(DEST + name)

def clip_X(I, x):
    return min(I.shape[1]-1,max(x,0))

def clip_Y(I, y):
    return min(I.shape[0]-1,max(y,0))

def pix(I, y, x):
    return I[clip_Y(I,y), clip_X(I,x)]

def union_box(B1, B2):
    # two boxes [[BOTTOM LEFT POINT], [TOP RIGHT POINT]]
    return [[min(B1[0][0], B2[0][0]), min(B1[0][1], B2[0][1])],
            [max(B1[1][0], B2[1][0]), max(B1[1][1], B2[1][1])]]

def compute_box(imShape, H):
  points = [[imShape[0], .0], [.0, imShape[1]], [imShape[0], imShape[1]]]
  y, x, w = H.dot(np.array([0.0, 0.0, 0.0]))
  # +1. for avoid dividing by 0
  ymin = ymax = int(round((y)/(w+1.)))
  xmin = xmax = int(round((x)/(w+1.)))
  for point in points:
    newY, newX, newW = H.dot(np.array(point + [1.]))
    newY, newX = (int(round(newY / newW)), int(round(newX / newW)))
    ymin, ymax, xmin, xmax = (min(ymin, newY), max(ymax, newY), min(xmin, newX), max(xmax, newX))
  return [[ymin, xmin], [ymax, xmax]]

def translate(bbox):
    out = np.identity(3)
    out[0, 2] = -bbox[0][0]
    out[1, 2] = -bbox[0][1]
    return out

def add_equations(M, i, coords):
    # modify M in place, does not need to return
    y, x  = (coords[0][0], coords[0][1])
    yp,xp = (coords[1][0], coords[1][1])
    M[2*i] = np.array([y, x, 1, 0, 0, 0, -y*yp, -x*yp, -yp])
    M[2*i+1] = np.array([0, 0, 0, y, x, 1, -y*xp, -x*xp, -xp])

