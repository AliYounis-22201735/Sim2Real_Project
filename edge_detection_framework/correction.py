import cv2
import numpy as np

# Taken from https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df

t = np.arange(256)

_lut_cache = dict()


def get_LUT(shadow_gain: float, highlight_gain: float):
    # Tone LUT
    key = f"{shadow_gain}, {highlight_gain}"

    if key not in _lut_cache:
        LUT_shadow = (1 - np.power(1 - t / 255, shadow_gain)) * 255
        LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + 0.5)))
        LUT_highlight = np.power(t / 255, highlight_gain) * 255
        LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + 0.5)))
        _lut_cache[key] = (LUT_shadow, LUT_highlight)

    return _lut_cache[key]


def correction(
    img: np.ndarray,
    shadow_amount_percent: float,
    shadow_tone_percent: float,
    shadow_radius: int,
    highlight_amount_percent: float,
    highlight_tone_percent: float,
    highlight_radius: int,
):
    """Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP

    Args:
        img (np.ndarray): input RGB image numpy array of shape (height, width, 3)
        shadow_amount_percent (float)[0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
        shadow_tone_percent (float)[0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
        shadow_radius (int)[>0]: Controls the size of the local neighborhood around each pixel
        highlight_amount_percent (float)[0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
        highlight_tone_percent (float)[0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
        highlight_radius (int)[>0]: Controls the size of the local neighborhood around each pixel

    Returns:
        np.ndarray: colour corrected image
    """
    shadow_tone = (shadow_tone_percent + 1e-6) * 255
    highlight_tone = 255 - highlight_tone_percent * 255

    shadow_gain = 1 + shadow_amount_percent * 6
    highlight_gain = 1 + highlight_amount_percent * 6

    # extract height, width
    height, width = img.shape[:2]

    # The entire correction process is carried out in YUV space,
    # adjust highlights/shadows in Y space, and adjust colors in UV space
    # convert to Y channel (grey intensity) and UV channel (color)
    img_YUV = cv2.cvtColor(img.astype("float32"), cv2.COLOR_BGR2YUV)
    img_Y, img_U, img_V = (img_YUV[..., x].reshape(-1) for x in range(3))

    # extract shadow / highlight
    shadow_map = 255 - (img_Y * 255) / shadow_tone
    shadow_map[np.where(img_Y >= shadow_tone)] = 0

    highlight_map = 255 - (255 * (255 - img_Y)) / (255 - highlight_tone)
    highlight_map[np.where(img_Y <= highlight_tone)] = 0

    # Gaussian blur on tone map, for smoother transition
    if shadow_amount_percent * shadow_radius > 0:
        shadow_map = cv2.GaussianBlur(
            shadow_map.reshape(height, width), (shadow_radius, shadow_radius), sigmaX=0
        ).reshape(-1)

    if highlight_amount_percent * highlight_radius > 0:
        highlight_map = cv2.GaussianBlur(
            highlight_map.reshape(height, width),
            (highlight_radius, highlight_radius),
            0,
        ).reshape(-1)

    # Tone LUT
    LUT_shadow, LUT_highlight = get_LUT(shadow_gain, highlight_gain)

    # adjust tone
    shadow_map /= 255
    highlight_map /= 255

    iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
    img_Y = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]

    # re convert to RGB channel
    img_YUV = (
        np.row_stack([img_Y, img_U, img_V]).T.reshape(height, width, 3).astype("float32")
    )
    output = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    output = np.maximum(0, np.minimum(output, 255)).astype(np.uint8)
    return output
