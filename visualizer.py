from PIL import Image, ImageColor

# import matplotlib.colors as mcolors
import numpy as np

# from toolbox.mask_encoding import b64_mask_decode
# from toolbox.img_utils import im_draw_bbox, im_draw_point, im_color_mask


def mask_to_xyxy(mask: np.ndarray, verbose: bool = False) -> tuple:
    """Convert a binary mask of shape (h, w) to
    xyxy bounding box format (top-left and bottom-right coordinates).
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        if verbose:
            logger.warning("mask_to_xyxy: No object found in the mask")
        return None
    x_min = np.min(xs)
    y_min = np.min(ys)
    x_max = np.max(xs)
    y_max = np.max(ys)
    xyxy = (x_min, y_min, x_max, y_max)
    xyxy = tuple([int(i) for i in xyxy])
    return xyxy


def annotate_detections(
    im: Image.Image,
    l_obj: list,
    color_key: str = "class",
    bbox_width: int = 1,
    label_key: str = "object_id",
    color_dict: dict = {},
):
    # color_list is  a list of tuple(name, color_hex)
    color_list = list(
        mcolors.XKCD_COLORS.items()
    )  # list(mcolors.TABLEAU_COLORS.items())
    unique_color_keys = list(
        set([o[color_key] for o in l_obj if color_key in o.keys()])
    )

    for obj in l_obj:
        color_index = unique_color_keys.index(obj[color_key])
        bbox_color = (
            color_dict[obj[color_key]] if color_dict else color_list[color_index][1]
        )
        im = (
            im_draw_bbox(
                im,
                color=bbox_color,
                width=bbox_width,
                caption=(str(obj[label_key]) if label_key else None),
                **obj["boundingBox"],
                use_bbv=True,
            )
            if "boundingBox" in obj.keys()
            else im_draw_point(
                im,
                **obj["point"],
                width=bbox_width,
                caption=(str(obj[label_key]) if label_key else None),
                color=bbox_color,
            )
        )
    return im


def annotate_masks(
    im: Image.Image, masks: list, mask_alpha: float = 0.9, bbox_width: int = 3
) -> Image.Image:
    """returns an annotated pillow image"""
    masks = [
        b64_mask_decode(m).astype(np.uint8) if isinstance(m, str) else m for m in masks
    ]
    segs = []
    for i, m in enumerate(masks):
        x0, y0, x1, y1 = mask_to_xyxy(m)
        segs.append(
            {
                "object_id": i,
                "boundingBox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            }
        )
    ann_im = np.array(im)
    for i, m in enumerate(masks):
        m_color = list(mcolors.XKCD_COLORS.items())[i]
        ann_im = im_color_mask(
            ann_im,
            mask_array=m,
            alpha=mask_alpha,
            rbg_tup=ImageColor.getrgb(m_color[1]),
        )
    ann_im = annotate_detections(
        ann_im,
        l_obj=segs,
        color_key="object_id",
        label_key="object_id",
        bbox_width=bbox_width,
    )
    return ann_im
