import numpy as np
import lab4
from utils import *

def apply_homography_test_factor(name, pathA, pathB, H, Bilinear=True):
  print("TEST: " + name + " : apply homography ")
  IA = read_img(pathA)
  IB = read_img(pathB)
  out = lab4.apply_homography_robust(IB, IA, H, Bilinear)
  dest = name + "_apply_homography.png"
  save_img(out, dest)
  print("  " + str(dest) + " saved!")


def compute_and_apply_homography_test_factor(name, pathA, pathB, pListA, pListB, Bilinear=True):
  print("TEST: " + name + " : compute and apply homography")
  IA = read_img(pathA)
  IB = read_img(pathB)
  h, w = IB.shape[0]-1, IB.shape[1]-1
  pointListT=[np.array([170, 95, 1]), np.array([171, 238, 1]), np.array([233, 235, 1]), np.array([239, 94, 1])]

  listOfPairs = zip(pListA, pListB)
  H = lab4.compute_homography(listOfPairs)
  IB = IB * 0.8
  out = lab4.apply_homography(IA, IB, H, Bilinear)
  dest = name + "_compute_and_apply_homography.png"
  save_img(out, dest)
  print("  " + str(dest) + " saved!")

def boxing_test_factor(name, pathA, pathB, pListA, pListB, Bilinear=True):
  print("TEST: " + name + " boxing")
  IA = read_img(pathA)
  IB = read_img(pathB)
  ([np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])], [np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])])
  listOfPairs=zip(pListA, pListB)
  out = lab4.stitch(IA, IB, listOfPairs)
  dest = name + "_boxing.png"
  save_img(out, dest)
  print("  " + str(dest) + " saved!")



apply_homography_test_factor("tram", 'data/green.png', 'data/poster.png', H=np.array([[1.12265192e+00, 1.44940136e-01, 1.70000000e+02], [8.65164180e-03, 1.19897030e+00, 9.50000000e+01],[  2.55704864e-04, 8.06420365e-04, 1.00000000e+00]]) , Bilinear=True)

compute_and_apply_homography_test_factor("tram", "data/poster.png", "data/green.png", [np.array([0, 0, 1]), np.array([0, 143, 1]), np.array([66, 143, 1]), np.array([66, 0, 1])], [np.array([170, 95, 1]), np.array([171, 238, 1]), np.array([233, 235, 1]), np.array([239, 94, 1])])

compute_and_apply_homography_test_factor("stata", "data/stata-1.png", "data/stata-2.png",  [np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])], [np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])])

boxing_test_factor("stata", "data/stata-1.png", "data/stata-2.png",  [np.array([209, 218, 1]), np.array([425, 300, 1]), np.array([209, 337, 1]), np.array([396, 336, 1])], [np.array([232, 4, 1]), np.array([465, 62, 1]), np.array([247, 125, 1]), np.array([433, 102, 1])])

boxing_test_factor("science", "data/science-1.jpg", "data/science-2.jpg", [np.array([192, 45, 1]), np.array([117, 146, 1]), np.array([338, 129, 1]), np.array([303, 33, 1])],[np.array([185, 230, 1]), np.array([93, 332, 1]), np.array([321, 323, 1]), np.array([293, 226, 1])])

