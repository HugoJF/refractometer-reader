import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MAX_SEARCH_AREA_RATIO = 0.1

BLUR_KERNEL_SIZE = 21
GOOD_MATCH_RATIO = 0.7
SCALE_MAX_Y = 260
SCALE_MAX_Y_VALUE = 0
SCALE_MIN_Y = 890
SCALE_MIN_Y_VALUE = 10
STRIP_START_X = 350
STRIP_END_X = 420
USE_MEDIAN = False

fig, axs = plt.subplots(4)
fig.tight_layout()
current_axs = 0


def next_axs():
    global current_axs

    ax = axs[current_axs]
    current_axs = current_axs + 1

    return ax


def calculate_value(input, reference, show_plots=True, write_images=True, debug_enabled=True):
    def debug(*args):
        if debug_enabled:
            print(*args)

    # read images
    img = cv.imread(input, cv.IMREAD_GRAYSCALE)  # queryImage
    reference = cv.imread(reference, cv.IMREAD_GRAYSCALE)  # trainImage

    # normalize images
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    reference = cv.normalize(reference, None, 0, 255, cv.NORM_MINMAX)

    # initialize SIFT
    fd = cv.SIFT_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = fd.detectAndCompute(img, None)
    kp2, des2 = fd.detectAndCompute(reference, None)

    # initialize matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # run matcher
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for maxima, n in matches:
        if maxima.distance < GOOD_MATCH_RATIO * n.distance:
            good.append([maxima])

    # draw matches
    img3 = cv.drawMatchesKnn(img, kp1, reference, kp2, good, None,
                             flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if write_images:
        cv.imwrite('matches-side-by-side.jpeg', img3)

    # rebuild arrays of good points
    points1 = np.float32([kp1[m.queryIdx].pt for m in np.asarray(good).flatten()]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in np.asarray(good).flatten()]).reshape(-1, 1, 2)

    # find homography
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)

    # use homography
    height, width = reference.shape
    img4 = cv.warpPerspective(img, h, (width, height))
    if write_images:
        cv.imwrite('post-homography.jpeg', img4)

    # crop test strip
    start = (STRIP_START_X, SCALE_MAX_Y)
    end = (STRIP_END_X, SCALE_MIN_Y)
    img4 = img4[start[1]:end[1], start[0]:end[0]]
    if write_images:
        cv.imwrite('test-strip.jpeg', img4)

    # rotate strip and plot it
    if show_plots:
        test_strip = next_axs()
        test_strip.set_title('Raw Strip')
        test_strip.imshow(np.rot90(img4, 1), extent=[SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE, 0, img4.shape[0]],
                          aspect='auto')

    # average out the whole strip
    if USE_MEDIAN:
        img4 = np.median(img4, axis=1)
    else:
        img4 = np.average(img4, axis=1)
    plot = img4.flatten()

    def convert_index_to_ratio(index, invert=False):
        ratio = index / len(plot)

        if invert:
            return 1 - ratio
        else:
            return ratio

    # currently we are talking about salinity
    def convert_index_to_value(index, invert=False):
        ratio = convert_index_to_ratio(index)
        delta = SCALE_MAX_Y_VALUE - SCALE_MIN_Y_VALUE

        if invert:
            return SCALE_MAX_Y_VALUE - (delta * ratio)
        else:
            return SCALE_MIN_Y_VALUE + (delta * ratio)

    # plot it
    if show_plots:
        x = np.linspace(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE, len(plot))
        tickcount = round(np.ceil(abs(SCALE_MAX_Y_VALUE - SCALE_MIN_Y_VALUE) + 1))
        labels = np.linspace(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE, tickcount, dtype=int)
        averaged_strip = next_axs()
        averaged_strip.plot(x, plot)
        averaged_strip.set_title('Averaged Strip')
        averaged_strip.set_xlim(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE)
        averaged_strip.set_xticks(labels)  # add ticks
        averaged_strip.set_xticklabels(labels)  # add ticks

    # convert plot to float32 (better gaussian blur results)
    plot = np.float32(plot)

    # smooth out plot
    plot = cv.GaussianBlur(plot, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0).flatten()

    # plot it
    if show_plots:
        x = np.linspace(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE, len(plot))
        tickcount = round(np.ceil(abs(SCALE_MAX_Y_VALUE - SCALE_MIN_Y_VALUE) + 1))
        labels = np.linspace(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE, tickcount, dtype=int)
        raw = next_axs()
        raw.plot(x, plot)
        raw.set_title('Smoothed')
        raw.set_xlim(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE)
        raw.set_xticks(labels)  # add ticks
        raw.set_xticklabels(labels)  # add ticks

    # derivative
    plot2 = np.diff(plot)

    # plot it
    if show_plots:
        x = np.linspace(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE, len(plot2))
        tickcount = round(np.ceil(abs(SCALE_MAX_Y_VALUE - SCALE_MIN_Y_VALUE) + 1))
        labels = np.linspace(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE, tickcount, dtype=int)
        derivative = next_axs()
        derivative.plot(x, plot2)
        derivative.set_title('Derivative')
        derivative.set_xlim(SCALE_MIN_Y_VALUE, SCALE_MAX_Y_VALUE)
        derivative.set_xticks(labels)  # add ticks
        derivative.set_xticklabels(labels)  # add ticks

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        # return array[idx]
        return idx

    raw_min = min(plot)
    raw_max = max(plot)
    half = (raw_max + raw_min) / 2

    maxima = plot2.argmax()
    delta = len(plot) * MAX_SEARCH_AREA_RATIO

    start = int(max(maxima - delta, 0))
    end = int(min(maxima + delta, len(plot)))
    halfcross = find_nearest(plot[start:end], half)
    halfcross = halfcross + start

    via_argmax = plot2.argmax()
    via_middle = halfcross

    debug("value (argmax)", convert_index_to_ratio(via_argmax))
    debug("value (middle)", convert_index_to_ratio(via_middle))

    # draw vertical line on all axs
    if show_plots:
        for ax in axs:
            via_argmax_value = convert_index_to_value(via_argmax)
            via_argmax_label = 'argmax: {:.2f}'.format(via_argmax_value)
            ax.axvline(x=via_argmax_value, color='r', linestyle='--', label=via_argmax_label)

            via_middle_value = convert_index_to_value(via_middle)
            via_middle_label = 'middle: {:.2f}'.format(via_middle_value)
            ax.axvline(x=via_middle_value, color='b', linestyle='--', label=via_middle_label)

            ax.legend()

        # remove x-axis margins
        for ax in axs:
            ax.margins(x=0)

        plt.show()

    return convert_index_to_ratio(via_argmax), convert_index_to_ratio(via_middle)


if __name__ == '__main__':
    images = [
        # 'in.jpeg',
        # 'in2.jpeg',
        # 'in3.jpeg',
        # 'inz.jpeg',
        'beer3_afterfermenting1.jpg',
        'beer3_afterfermenting2.jpg',
        'beer3_afterfermenting3.jpg',
        'beer3_afterfermenting4.jpg',
    ]

    for image in images:
        argmax, middle = calculate_value(image, 'crop.jpeg', show_plots=True, write_images=False, debug_enabled=True)

        # print tds with 2 decimal places
        print(f'{image}: argmax={argmax:.2f} | middle={middle:.2f}')
