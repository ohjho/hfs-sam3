import base64, os, io, random, time
from PIL import Image
import numpy as np

def b64_mask_encode(mask_np_arr, tmp_dir = '/tmp/miro/mask_encoding/'):
    '''
    turn a binary mask in numpy into a base64 string
    '''
    mask_im = Image.fromarray(np.array(mask_np_arr).astype(np.uint8)*255)
    mask_im = mask_im.convert(mode = '1') # convert to 1bit image

    if not os.path.isdir(tmp_dir):
        print(f'b64_mask_encode: making tmp dir for mask encoding...')
        os.makedirs(tmp_dir)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    hash_str = random.getrandbits(128)
    tmp_fname = tmp_dir + f'{timestr}_{hash_str}_mask.png'
    mask_im.save(tmp_fname)
    return base64.b64encode(open(tmp_fname, 'rb').read())

def b64_mask_decode(b64_string):
    '''
    decode a base64 string back to a binary mask numpy array
    '''
    im_bytes = base64.b64decode(b64_string)
    im_decode = Image.open(io.BytesIO(im_bytes))
    return np.array(im_decode)

def get_true_mask(mask_arr, im_w_h:tuple, x0, y0, x1, y1):
    '''
    decode the mask of CM output to get a mask that's the same size as source im
    '''
    if x0 > im_w_h[0] or x1 > im_w_h[0] or y0 > im_w_h[1] or y1 > im_w_h[1]:
        raise ValueError(f'get_true_mask: Xs and Ys exceeded im_w_h bound: {im_w_h}')

    if mask_arr.shape != (y1 - y0, x1 - x0):
        raise ValueError(f'get_true_mask: Bounding Box h: {y1-y0} w: {x1-x0} does not match mask shape: {mask_arr.shape}')

    w, h = im_w_h
    mask = np.zeros((h,w), dtype = np.uint8)
    mask[y0:y1, x0:x1] = mask_arr
    return mask
