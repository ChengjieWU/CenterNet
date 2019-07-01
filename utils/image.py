import cv2
import numpy as np
import random

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_(image, mean, std):
    image -= mean
    image /= std

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_jittering_(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)

def crop_image(image, center, size):
    """以center为中心，在image上切下大小为size的图片。

    保证center中心仍在crop图片的中心，且超出原图边缘的部分，补0。

    :param image: image
    :param center: (y, x) for center
    :param size: (height, width)
    :return: cropped_image: crop出的图片. border: [ytl, tbr, xtl, xbr], 是非0区域
             在crop出的图片中的坐标. offset: 截图左上角点在原图中的坐标。
    """
    cty, ctx            = center
    height, width       = size
    im_height, im_width = image.shape[0:2]
    cropped_image       = np.zeros((height, width, 3), dtype=image.dtype)

    # 由于原图大小限制，实际截到的位置
    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    # 实际截到的图，边缘距离crop中心的距离
    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    # 保证center中心仍在crop出图片的中心
    cropped_cty, cropped_ctx = height // 2, width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - height // 2,
        ctx - width  // 2
    ])

    return cropped_image, border, offset


def constraint_max_size(x, cons: int):
    """限制图片最长边不超过某一值

    :param x: A 3-d numpy array of shape [height, width, channel].
    :param cons: max constraint
    :return: A resized 3-d numpy array.
    """
    height, width = x.shape[0:2]
    if max(height, width) <= cons:
        return x
    if height > width:
        ratio = height / cons
        # cv2这边很坑，size为先横轴再纵轴，与numpy的顺序是反的
        return cv2.resize(x, (int(width / ratio), cons))
    else:
        ratio = width / cons
        return cv2.resize(x, (cons, int(height / ratio)))


def pad_resize_image(x, size):
    """Pad and resize a given image.

    :param x: A 3-d numpy array of shape [height, width, channel].
    :param size: (height, width)
    :return: A padded and resized 3-d numpy array.
    """
    height, width = size

    if x.shape[0] > x.shape[1]:
        diff = x.shape[0] - x.shape[1]
        x = cv2.copyMakeBorder(x, 0, 0, diff // 2, diff - diff // 2,
                               borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        diff = x.shape[1] - x.shape[0]
        x = cv2.copyMakeBorder(x, diff // 2, diff - diff // 2, 0, 0,
                               borderType=cv2.BORDER_CONSTANT, value=0)
    x = cv2.resize(x, (height, width),
                   interpolation=cv2.INTER_LINEAR)
    return x
