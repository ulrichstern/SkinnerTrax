#
# determine chamber position in image by template matching
#
# 27 Nov 2012 by Ulrich Stern
#
# notes:
# * params:
#     image file or, for test mode, directory or file with image filenames,
#     template name (optional, default: "Standard"),
#     skip showing match (0|1, optional, default: 0)
# * returns (comma-joined for MATLAB)
#   - position of template in image (x, y) (if submatch used, for top match)
#   - minimum distance (x or y) of template to image border
#   - position of template in image (x, y) for bottom match (only if submatch
#      used) or resize factor (only if resize used)
# * sample calls:
#   p27 match_template.py C:/SSD/tracking/Uli/match_test.jpg
#   p27 match_template.py T:/GB Long720p
#   python match_template.py smplStandard.jpg
#   python match_template.py smplLgC.jpg LgC
#

import sys, os
import cv2
import numpy as np
import shutil
import math
import collections

from util import *

# - - -

SHOW_INTERNAL_IMAGES = False
SHOW_MAIN_MATCH_ALSO = False   # if submatch used

DEBUG = False

# constants below usually not modified

COPY_IMAGES = False   # False for normal operation
COPY_DST = "C:/Users/Tracking/Videos/tracking/analysis tests/" + \
  "template matching/Standard Synology Uli/"
EXCLUDE_FROM_COPY = re.compile(r'[/\\]orig p[/\\]', re.IGNORECASE)

IMG_HEIGHT = 480   # used when searching directory tree

TEST_DIR = "__match_template_test__"

DFLT_TMPLT = "Standard"
TMPLT_PATH = os.environ.get('YL_TMPLT_PATH') or "C:/SSD/tracking/Ctrax"
TMPLT_PREFIX = os.path.join(TMPLT_PATH, "tmplt")
TMPLTS = {
  # note: thr2 should be tuned to produce good edge image for template
  'Standard': {
    'rotate': False,
    'edge': True,
    'resize': [280./268, 278./268, 276./268, 274./268, 272./268, 270./268,
      266./268, 264./268, 262./268, 260./268, 258./268, 256./268, 254./268,
      252./268, 250./268, 248./268],
    'thr2': 120,
    'lines': [[(41,5), (351,5)], [(41,275), (351,275)],
      [(174,22), (174,252)], [(218,22), (218,252)]],
    # note: submatch had problems with the UV LED side sometimes
    'submatch_NONE': [ {
      'x1y1x2y2d': [3,2, 389,102, 10],   # top
      'lines': [[(41,6), (351,6)],
        [(174,20), (174,52)], [(218,20), (218,52)]]
      }, {
      'x1y1x2y2d': [3,179, 389,279, 10],   # bottom
      'lines': [[(41,275), (351,275)],
        [(174,229), (174,261)], [(218,229), (218,261)]]
      } ]
  },
  'Long720p': {
    'rotate': True,
    'edge': False,
    'thr2': 240,
    'submatch': [ {
      'x1y1x2y2d': [36,101, 281,140, 10],   # top
      'lines': [[(56,105), (262,105)],
        [(145,112), (145,141)], [(172,112), (172,141)]]
      }, {
      'x1y1x2y2d': [34,819, 281,858, 10],   # bottom
      'lines': [[(52,853), (262,853)],
        [(141,813), (141,845)], [(170,813), (170,845)]]
      } ]
  },
  'HtL' : {
    'rotate': False,
    'edge': True,
    'resize': [
      668./654, 666./654, 664./654, 662./654, 660./654, 658./654, 656./654,
      652./654, 650./654, 648./654, 646./654, 644./654, 642./654, 640./654],
    'thr2': 80,
    'lines': [[(5,4), (659,4)], [(5,658), (659,658)],
      [(5,4), (5,658)], [(659,4), (659,658)]]
  },
  'LgC' : {
    'rotate': False,
    'edge': True,
    'resize': [538./528, 536./528, 534./528, 532./528, 530./528,
      526./528, 524./528, 522./528, 520./528, 518./528],
    'thr2': 80,
    'lines': [[(4,4), (532,4)], [(532,4), (532,532)],
      [(4,4), (4,532)], [(4,532), (532,532)]]
  }
}
TXT_STL_BK = (cv2.FONT_HERSHEY_PLAIN, 1, COL_BK, 1, 8)
TXT_STL_W = (cv2.FONT_HERSHEY_PLAIN, 1, COL_W, 1, 8)
TXT_STL_Y = (cv2.FONT_HERSHEY_PLAIN, 1, COL_W, 1, 8)
TXT_STL_CMDS = (cv2.FONT_HERSHEY_PLAIN, .9, COL_BK, 1, CV_AA)

SCRIPT = __name__ == "__main__"

# - - -

# draws the given lines on the given image
def drawLines(img, lines, ptMap, color):
  for line in lines:
    p1, p2 = line
    cv2.line(img, ptMap(p1), ptMap(p2), color, 1)

# adds edge images to the given mtargs
def addEdgeImages(mtargs, img, tmpl, test, prfx='', rszFctr=None, dilate=None,
                  submatch=False, thr2=120):
  # template
  tmplE = edgeImg(tmpl, thr2=thr2)[0]
  if dilate is not None:
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate,dilate))
    tmplE = cv2.dilate(tmplE, kern)
  if SHOW_INTERNAL_IMAGES and not test:
    imshow(prfx+('resize %.3f '%rszFctr if rszFctr else '')+"tmplE", tmplE)

  # images
  neis = 0
  thrs = [150, 100, 75] if submatch else \
    [150, 100, 75, (10,50), (10,30), (5, 15)]
    # note: (5,15) for UV case (p3__2013-02-11__18-54-18) w/out normalization
  for thr in thrs:
    thr1, thr2 = thr if isinstance(thr, tuple) else (20, thr)
    imgE, nz = edgeImg(img, thr1=thr1, thr2=thr2)
    hlns = cv2.HoughLinesP(imgE, 1, math.pi/2, 100, None, 100, 0)
      # note: math.pi/2 guarantees lines are horizontal or vertical
      #  previously used: ..., 150, None, 80, 0
    lns = [] if hlns is None else hlns.squeeze(   # works for OpenCV 2 and 3
      axis=0 if hlns.shape[0] == 1 else 1)
    ch = cv = 0
    for ln in lns:
      x1, y1, x2, y2 = ln
      ch, cv = ch + (y1 == y2), cv + (x1 == x2)
    if DEBUG and not test and rszFctr is None:
      print prfx + '%s: nz=%.3f, ch=%d, cv=%d' %(thr2, nz, ch, cv)
    if not submatch and (nz < 0.005 or ch+cv < 2) \
       or submatch and nz < 0.01:
      # note: skipping more seems to lead to worse results w/ or w/out
      #  normalization
      continue

    if SHOW_INTERNAL_IMAGES and not test and rszFctr is None:
      imgEC = overlay(img, imgE)
      for idx, ln in enumerate(lns):
        x1, y1, x2, y2 = ln
        cv2.line(imgEC, (x1, y1), (x2, y2), COL_Y, 1)
        cv2.putText(imgEC, str(idx), (x1, y1), *TXT_STL_Y)
      imshow(prfx+"imgEC %d"%thr2, imgEC)
    mtargs.extend([imgE, tmplE])
    neis += 1

  if neis < 2 and test and rszFctr is None:
    warn("number of edge images: %d"%neis)

# put text for template
def putText(img, txt, tlx, tly, tmpl, style=TXT_STL_BK):
  wh = textSize(txt, style)
  cv2.putText(img, txt, (tlx+5,tly+(tmpl.shape[0]+wh[1])/2), *style)

# - - -

# match on a subimage of the template
def submatch(img, tmpl, sm, off, thr2, test):
  prfx = "sub %d: " %off[1]

  mtargs = [img, tmpl]
  if not 'edge' in sm or sm['edge']:
    # note: edge detection used by default since fly can have bigger impact
    #  for submatch (and edge detection reduces impact)
    addEdgeImages(mtargs, img, tmpl, test, prfx, submatch=True, thr2=thr2,
      dilate=3)

  res, tlx, tly, br, minD, val, vals = matchTemplate(*mtargs)
  if SHOW_INTERNAL_IMAGES and not test:
    showNormImg(prfx+"res", res)
  if DEBUG and not test:
    print "vals:", vals

  def ptMap(pt): return tupleAdd((tlx, tly), tupleSub(pt, off))
  drawLines(img, sm['lines'], ptMap, COL_BK)
  putText(img, "%.2f"%val, tlx, tly, tmpl)
  return tlx, tly

# - - -

def showMatch(img, txt=None, footer=True):
  if img is None:
    return
  txt = ['0: frame 0 | 1: first 100', 'r: random', 'background: b: 10 | B: 20',
    's: save | q: quit (no save)'] if txt is None else [txt]
  if footer:
    wht, nl = textSize(txt[0], TXT_STL_CMDS), len(txt)
    h, w = img.shape[:2]
    img = extendImg(img, (0, 0, wht[1]*nl + 8*(nl-1) + 16, 0))
    for i, t in enumerate(txt):
      util.putText(img, t, (w/2, h+7+(wht[1]+8)*i), (-.5, 1), TXT_STL_CMDS)
  cv2.imshow("template match", img)
  cv2.waitKey(1)

# returns image to match template against and whether to show menu
def pickImage(imgF, key, img):
  if os.path.isfile(imgF):
    return cv2.imread(imgF, 1), False
  mf = replaceCheck(JPG_X, ".avi", imgF)
  mf = replaceCheck(r'([\/])p(\d)__([^\/]+)$', r'\1c\2__\3', mf)
  if not os.path.isfile(mf):
    error("movie file does not exist")
  cap = cv2.VideoCapture(mf)
  nf, fps = frameCount(cap), frameRate(cap)
  while True:
    ns = 20 if key == 'B' else (10 if key == 'b' else 1)
    frmIdxs = np.sort(np.random.choice(nf, min(nf, ns), replace=False)) \
      if key not in {'0', '1'} else range(100 if key == '1' else 1)
    frms, ns = [], len(frmIdxs)
    for i, frmIdx in enumerate(frmIdxs):
      showMatch(img, 'please wait (reading frame %d of %d)' %(i+1, ns))
      try:
        frms.append(readFrame(cap, frmIdx))
      except util.VideoError:
        pass
    if frms:
      break
  cap.release()
  ns = len(frms)
  k = ns/2 if ns < 5 else np.random.randint(.2*ns, .8*ns)
  frms = np.partition(np.array(frms), k, axis=0)
  showMatch(img, ('' if k == 0 else 'using k=%d, ' %k) +
    'please wait (matching)')
  return frms[k], True

# - - -

# match on main (or full) template
# notes:
# * imgF can also be used to pass image
# * return values are generally integer; factor is float only if it is not 1
def match(imgF, tmplN=None, show=False, test=False, outF=None):
  if tmplN is None:
    tmplN = DFLT_TMPLT
  td = TMPLTS.get(tmplN)
  if td is None:
    error("template name %s unknown" %tmplN)
  tf = TMPLT_PREFIX+tmplN+'.jpg'
  checkIsfile(tf)
  thr2 = td.get('thr2') or 120

  key, img, menu = '0', None, False
  while True:
    if isinstance(imgF, np.ndarray):
      img = toColor(imgF)
    else:
      img, menu = pickImage(imgF, key, img)
      imgO = img.copy()
    if td['rotate']:
      img = cv2.transpose(img)
      img = cv2.flip(img, 1)
    tmpl = cv2.imread(tf, 1)

    # preprocess images
    img, tmpl = [normalize(i) for i in [img, tmpl]]
    img, tmpl = [cv2.GaussianBlur(i, (3, 3), 0) for i in [img, tmpl]]
    tmpl1 = tmpl.copy()   # tmpl may get resized

    def mtargs(rszFctr=None):
      args = [img, tmpl]
      if td['edge']:
        addEdgeImages(args, img, tmpl, test, rszFctr=rszFctr, dilate=4,
          thr2=thr2)
        # note: dilate 4 was required for p2__2013-09-12__19-17-34
      return args

    # match
    result, tlx, tly, br, minD, val, vals = matchTemplate(*mtargs())
    fctrs = td.get('resize') or []
    fctr, maxSV = 1, sum(vals[1:])
    for f in fctrs:
      tmpl = cv2.resize(tmpl1, (0,0), fx=f, fy=f)
      vs = tuple(matchTemplate(*mtargs(f)))
      sv = sum(vs[6][1:])
      assert sv > 0
      if DEBUG and not test:
        print "factor %.3f: %.3f (max %.3f)" %(f, sv, maxSV)
      if sv > maxSV:
        result, tlx, tly, br, minD, val, vals = vs
        maxSV, fctr = sv, f
    if DEBUG and fctr != 1:
      print "choosing resize (%.3f)"%fctr
    if SHOW_INTERNAL_IMAGES and not test:
      showNormImg("result", result)

    # submatch
    def ptMap(pt): return tupleAdd((tlx, tly), intR(tupleMul(pt, fctr)))
    if 'submatch' in td:
      tlSm = []
      for sm in td['submatch']:
        x1, y1, x2, y2, d = sm['x1y1x2y2d']
        stlx, stly = submatch(img[tly+y1-d:tly+y2+d, tlx+x1-d:tlx+x2+d],
          tmpl1[y1:y2, x1:x2], sm, (x1, y1), thr2, test)
        tlSm.append([tlx+stlx-d, tly+stly-d])

        if SHOW_MAIN_MATCH_ALSO:
          drawLines(img, sm['lines'], ptMap, COL_Y)
    else:
      if 'lines' in td:
        wht = np.mean(img) < 128
        drawLines(img, td['lines'], ptMap, COL_W if wht else COL_BK)
        putText(img, "%.2f"%val, tlx, tly, tmpl,
          TXT_STL_W if wht else TXT_STL_BK)

    # template match image
    #cv2.rectangle(img, (tlx, tly), br, COL_W, 1)
    d = 10
    img = subimage(img, (tlx-d,tly-d), tupleAdd(br,d))
    if test:
      cv2.imwrite(TEST_DIR+'/'+os.path.split(imgF)[1], img)
    elif show:
      showMatch(img, footer=menu)
    if outF:
      cv2.imwrite(outF, img)

    if SCRIPT and not test:
      if not menu:
        cv2.waitKey(0)
        break
      while True:
        key = chr(cv2.waitKey(0) & 255)
        if key in {'0', '1', 'r', 'b', 'B', 's', 'q'}:
          break
      if key in {'s', 'q'}:
        if key == 's':
          cv2.imwrite(imgF, imgO)
        break
    else:
      break

  # output for MATLAB (see comments at top for fields)
  if not test:
    if 'submatch' in td:
      rl = tlSm[0] + [minD] + tlSm[1]
    else:
      rl = [tlx, tly, minD]
      if fctrs: rl.append(fctr)
    if SCRIPT:
      sys.stdout.write(",".join(map(str, rl)))
    else:
      return rl

# - - -

# run match for
#  all images of height IMG_HEIGHT in given directory tree or
#  all image names in the given file
def test(treeOrFile, tmplN, show):
  print "=== test mode ==="
  if COPY_IMAGES:
    print "copying images"
  else:
    print "writing images w/ template matches into directory %s" %(TEST_DIR)
  if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
  if os.path.isfile(treeOrFile):
    for fn in open(treeOrFile):
      match(fn.strip(), tmplN, show, True)
  else:
    ns = collections.Counter()
    for dp, dns, fns in os.walk(treeOrFile):
      for fn in fns:
        if JPG_X.search(fn):
          fnf = os.path.join(dp, fn)
          if EXCLUDE_FROM_COPY.search(fnf):
            ns['regex'] += 1
            continue
          img = cv2.imread(fnf, 1)
          h = img.shape[0]
          if h == IMG_HEIGHT:
            print fnf
            if COPY_IMAGES:
              shutil.copyfile(fnf, os.path.join(COPY_DST, fn))
            else:
              match(fnf, tmplN, show, True)
          else:
            ns[h] += 1
    if ns:
      print "number of images skipped (by height or regex):"
      for k, c in sorted(ns.items()):
        print "  %s: %d" %(k, c)

# - - -

if SCRIPT:
  if len(sys.argv) == 1:
    error("at least one argument required (see comments for usage)")

  imgF = sys.argv[1]
  tmplN = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
  show = not (len(sys.argv) > 3 and int(sys.argv[3]))

  if JPG_X.search(imgF):
    match(imgF, tmplN, show)
  else:
    test(imgF, tmplN, show)

