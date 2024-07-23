import math

import cv2
import numpy as np
import torch
from scipy.interpolate import splprep, splev

import transform
from draw_gaussian import draw_umich_gaussian, gaussian_radius


def processing_test(image, input_h, input_w):
    image = cv2.resize(image, (input_w, input_h))
    out_image = image.astype(np.float32) / 255.0
    out_image = out_image - 0.5
    out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
    out_image = torch.from_numpy(out_image)
    return out_image


def draw_spinal(pts, out_image):
    colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
    for i in range(4):
        cv2.circle(out_image, (int(pts[i, 0]), int(pts[i, 1])), 3, colors[i], 1, 1)
        cv2.putText(
            out_image,
            "{}".format(i + 1),
            (int(pts[i, 0]), int(pts[i, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
            1,
        )
    for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0]):
        cv2.line(
            out_image,
            (int(pts[i, 0]), int(pts[i, 1])),
            (int(pts[j, 0]), int(pts[j, 1])),
            color=colors[i],
            thickness=1,
            lineType=1,
        )
    return out_image


def rearrange_pts(pts):
    # rearrange left right sequence
    boxes = []
    centers = []
    for k in range(0, len(pts), 4):
        pts_4 = pts[k : k + 4, :]
        x_inds = np.argsort(pts_4[:, 0])
        pt_l = np.asarray(pts_4[x_inds[:2], :])
        pt_r = np.asarray(pts_4[x_inds[2:], :])
        y_inds_l = np.argsort(pt_l[:, 1])
        y_inds_r = np.argsort(pt_r[:, 1])
        tl = pt_l[y_inds_l[0], :]
        bl = pt_l[y_inds_l[1], :]
        tr = pt_r[y_inds_r[0], :]
        br = pt_r[y_inds_r[1], :]
        # boxes.append([tl, tr, bl, br])
        boxes.append(tl)
        boxes.append(tr)
        boxes.append(bl)
        boxes.append(br)
        centers.append(np.mean(pts_4, axis=0))
    bboxes = np.asarray(boxes, np.float32)
    # rearrange top to bottom sequence
    centers = np.asarray(centers, np.float32)
    sort_tb = np.argsort(centers[:, 1])
    new_bboxes = []
    for sort_i in sort_tb:
        new_bboxes.append(bboxes[4 * sort_i, :])
        new_bboxes.append(bboxes[4 * sort_i + 1, :])
        new_bboxes.append(bboxes[4 * sort_i + 2, :])
        new_bboxes.append(bboxes[4 * sort_i + 3, :])
    new_bboxes = np.asarray(new_bboxes, np.float32)
    return new_bboxes


def generate_ground_truth(image, pts_2, image_h, image_w, img_id):
    hm = np.zeros((1, image_h, image_w), dtype=np.float32)
    wh = np.zeros((17, 2 * 4), dtype=np.float32)
    reg = np.zeros((17, 2), dtype=np.float32)
    ind = np.zeros((17), dtype=np.int64)
    reg_mask = np.ones((17), dtype=np.uint8)
    spline_mask = np.ones((100), dtype=np.uint8)
    cpie = np.zeros((17, 2 * 2), dtype=np.float32)
    avie = np.zeros((17, 2 * 4), dtype=np.float32)

    if pts_2[:, 0].max() > image_w:
        print("w is big", pts_2[:, 0].max())
    if pts_2[:, 1].max() > image_h:
        print("h is big", pts_2[:, 1].max())
    if pts_2.shape[0] != 68:
        print("ATTENTION!! image {} pts does not equal to 68!!! ".format(img_id))

    for k in range(17):
        pts = pts_2[4 * k : 4 * k + 4, :]
        bbox_h = np.mean(
            [
                np.sqrt(np.sum((pts[0, :] - pts[2, :]) ** 2)),
                np.sqrt(np.sum((pts[1, :] - pts[3, :]) ** 2)),
            ]
        )
        bbox_w = np.mean(
            [
                np.sqrt(np.sum((pts[0, :] - pts[1, :]) ** 2)),
                np.sqrt(np.sum((pts[2, :] - pts[3, :]) ** 2)),
            ]
        )
        cen_x, cen_y = np.mean(pts, axis=0)
        ct = np.asarray([cen_x, cen_y], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
        radius = max(0, int(radius))
        draw_umich_gaussian(hm[0, :, :], ct_int, radius=radius)
        ind[k] = ct_int[1] * image_w + ct_int[0]
        reg[k] = ct - ct_int
        for i in range(4):
            offset_x, offset_y = ct - pts[i, :]
            r = np.sqrt(offset_x**2 + offset_y**2)
            theta = np.arctan2(offset_y, offset_x)
            wh[k, 2 * i : 2 * i + 2] = [r, theta]
            # wh[k, 2 * i : 2 * i + 2] = ct - pts[i, :]

    pts_temp = pts_2.reshape(17, 4, 2).copy()
    centers = np.mean(pts_temp, axis=1)

    for idx, (center_prev, center_current, center_next) in enumerate(
        zip(centers[:, :], centers[1:, :], centers[2:, :])
    ):
        cpie[idx + 1, 0:2] = center_prev - center_current
        cpie[idx + 1, 2:4] = center_next - center_current

    cpie[0, 2:4] = -cpie[1, 0:2].copy()
    cpie[16, 0:2] = -cpie[15, 3:4].copy()

    for idx, (vetebra_prev, vetebra_current, vetebra_next) in enumerate(
        zip(pts_temp[:, :, :], pts_temp[1:, :, :], pts_temp[2:, :, :])
    ):
        avie[idx + 1, 0:4] = (vetebra_prev[2:4, :] - vetebra_current[0:2, :]).flatten()
        avie[idx + 1, 4:8] = (vetebra_next[0:2, :] - vetebra_current[2:4, :]).flatten()

    avie[0, 4:8] = -avie[1, 0:4].copy()
    avie[16, 0:4] = -avie[15, 4:8].copy()

    # Here I define a new indexes, it is still in between first ind and last ind
    # but with 100 points. We don't need to pay attention on what is the exact point
    # of those landmarks, since these indexes is only used on picking some points from
    # the feature maps of model outputs, and model doesn't care what you've picked; it
    # just fit it anyway.
    ind_new = np.linspace(ind[0], ind[-1], num=100, dtype=np.int64)

    def generate_b_spline_curve(pts, num_points=100):
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=0)
        unew = np.linspace(0, 1, num_points)
        out = splev(unew, tck)
        return np.array(out).T

    spline_left = generate_b_spline_curve(pts_2[::2, :])
    spline_mid = generate_b_spline_curve(centers)
    spline_right = generate_b_spline_curve(pts_2[1::2, :])

    spline = np.concatenate((spline_left, spline_mid, spline_right), axis=-1)

    ret = {
        "input": image,
        "hm": hm,
        "ind": ind,
        "reg": reg,
        "wh": wh,
        "ind_new": ind_new,
        "spline": spline,
        "spline_mask": spline_mask,
        "reg_mask": reg_mask,
        "cpie": cpie,
        "avie": avie,
    }

    return ret


# def filter_pts(pts, w, h):
#     pts_new = []
#     for pt in pts:
#         if any(pt) < 0 or pt[0] > w - 1 or pt[1] > h - 1:
#             continue
#         else:
#             pts_new.append(pt)
#     return np.asarray(pts_new, np.float32)


def processing_train(image, pts, image_h, image_w, down_ratio, aug_label, img_id):
    # filter pts ----------------------------------------------------
    h, w, c = image.shape
    # pts = filter_pts(pts, w, h)
    # ---------------------------------------------------------------
    data_aug = {
        "train": transform.Compose(
            [
                transform.ConvertImgFloat(),
                transform.PhotometricDistort(),
                transform.Expand(max_scale=1.5, mean=(0, 0, 0)),
                transform.RandomMirror_w(),
                transform.Resize(h=image_h, w=image_w),
            ]
        ),
        "val": transform.Compose(
            [transform.ConvertImgFloat(), transform.Resize(h=image_h, w=image_w)]
        ),
    }
    if aug_label:
        out_image, pts = data_aug["train"](image.copy(), pts)
    else:
        out_image, pts = data_aug["val"](image.copy(), pts)

    out_image = np.clip(out_image, a_min=0.0, a_max=255.0)
    out_image = np.transpose(out_image / 255.0 - 0.5, (2, 0, 1))
    pts = rearrange_pts(pts)
    pts2 = transform.rescale_pts(pts, down_ratio=down_ratio)

    return np.asarray(out_image, np.float32), pts2
