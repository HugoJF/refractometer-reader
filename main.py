import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MAX_SEARCH_AREA_RATIO = 0.1

BLUR_KERNEL_SIZE = 21
GOOD_MATCH_RATIO = 0.7
SCALE_MAX_Y = 260
SCALE_MIN_Y = 890
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


def calculate_tds(input, reference, show_plots=True, write_images=True, debug_enabled=True):
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
        test_strip.imshow(np.rot90(img4, 1))

    # average out the whole strip
    if USE_MEDIAN:
        img4 = np.median(img4, axis=1)
    else:
        img4 = np.average(img4, axis=1)
    plot = img4.flatten()

    # plot it
    if show_plots:
        averaged_strip = next_axs()
        averaged_strip.set_title('Averaged Strip')
        averaged_strip.plot(plot)

    # convert plot to float32 (better gaussian blur results)
    plot = np.float32(plot)

    # smooth out plot
    plot = cv.GaussianBlur(plot, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0).flatten()

    # plot it
    if show_plots:
        raw = next_axs()
        raw.plot(plot)
        raw.set_title('Smoothed')

    # derivative
    plot2 = np.diff(plot)

    # plot it
    if show_plots:
        derivative = next_axs()
        derivative.plot(plot2)
        derivative.set_title('Derivative')

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

    def convert_index_to_tds(index):
        ratio = index / len(plot)

        return 1 - ratio

    debug("tds (argmax)", convert_index_to_tds(via_argmax))
    debug("tds (middle)", convert_index_to_tds(via_middle))

    # draw vertical line on all axs
    if show_plots:
        for ax in axs:
            ax.axvline(x=via_argmax, color='r', linestyle='--')
            ax.axvline(x=via_middle, color='g', linestyle='--')

        # remove x-axis margins
        for ax in axs:
            ax.margins(x=0)

        plt.show()

    return convert_index_to_tds(via_argmax), convert_index_to_tds(via_middle)


if __name__ == '__main__':
    images = [
        'in.jpeg',
        'in2.jpeg',
        'in3.jpeg',
        'inz.jpeg',
    ]

    for image in images:
        tds1, tds2 = calculate_tds(image, 'crop.jpeg', show_plots=False, write_images=False, debug_enabled=False)

        # print tds with 2 decimal places
        print(f'{image}: {tds1:.2f} | {tds2:.2f}')
