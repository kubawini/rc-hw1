import cv2
import numpy as np
import matplotlib.pyplot as plt

### Task 1 ###
# def generate_obj_pts():
#     tag_side = 168
#     spacing = 70
#     rows = 2
#     cols = 3
#     coordinates = []
#     for i in range(cols):
#         for j in range(rows):
#             p1 = [i * (tag_side + spacing), j * (tag_side + spacing), 0]
#             p2 = [i * (tag_side + spacing), j * (tag_side + spacing) + tag_side, 0]
#             p3 = [i * (tag_side + spacing) + tag_side, j * (tag_side + spacing) + tag_side, 0]
#             p4 = [i * (tag_side + spacing) + tag_side, j * (tag_side + spacing), 0]
#             coordinates.append(p1)
#             coordinates.append(p2)
#             coordinates.append(p3)
#             coordinates.append(p4)
#     return np.array(coordinates, dtype=np.float32)
#
#
# def corners_to_img_pts(corners):
#     result = []
#     for item1 in corners:
#         for item2 in item1:
#             for item3 in item2:
#                 point = []
#                 for item4 in item3:
#                     point.append(item4)
#                 result.append([point])
#     result = np.array(result, dtype=np.float32)
#     return result
#
#
# def task1_logic(obj_points, img_points, size, img):
#     ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv2.calibrateCamera(
#         obj_points, img_points, size, None, None)
#     alpha = 1
#     rect_camera_mat = cv2.getOptimalNewCameraMatrix(camera_mat, distortion, size, alpha)[0]
#     map1, map2 = cv2.initUndistortRectifyMap(camera_mat, distortion, np.eye(3), rect_camera_mat, size,
#                                              cv2.CV_32FC1)
#     rect_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
#     rect_img_draw = rect_img.copy()
#     return rect_img_draw, rect_camera_mat, camera_mat, distortion, rotation_vecs, translation_vecs
#
#
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(dictionary, parameters)
# obj_pts = generate_obj_pts()
# size = (1280, 720)
# img1 = cv2.imread(f"calibration/img1.png")
#
# all_img_points_1 = []
# all_img_points_2 = []
# all_obj_points_1 = []
# all_obj_points_2 = []
#
# for i in range(1, 29):
#     img = cv2.imread(f"calibration/img{i}.png")
#     corners, ids, _ = detector.detectMarkers(img)
#     ids = np.array(ids.flatten())
#     permutation = np.argsort(ids)
#     sorted_corners = []
#     for p in reversed(permutation):
#         sorted_corners.append(corners[p])
#         all_img_points_2.append(np.array(corners[p]).reshape(4, 2))
#         all_obj_points_2.append(np.array([[0, 0, 0], [0, 168, 0], [168, 168, 0], [168, 0, 0]], dtype=np.float32))
#
#     sorted_corners_formatted = np.array(sorted_corners).reshape(24, 2)
#     all_img_points_1.append(sorted_corners_formatted)
#     all_obj_points_1.append(generate_obj_pts())
#
# rect_img_draw_1, rect_mat_1, _, _, _, _ = task1_logic(all_obj_points_1, all_img_points_1, size, img1)
# # rect_img_draw_2, rect_mat_2, _, _, _, _ = task1_logic(all_obj_points_2, all_img_points_2, size, img1)
#
# # Show the image
# cv2.imshow('img1', rect_img_draw_1)
# cv2.waitKey(0)

# Show the image
# cv2.imshow('img1', rect_img_draw_1)
# cv2.waitKey(0)

'''
Conclusion:
Sprawdzić projection error (albo jakąś własną modyfikację, bo być może projection error oszukuje)
'''


### Task 2 ###
def transform_shape(x_cart, y_cart, transformation_matrix):
    v = np.array([[x_cart, y_cart, 1], [0, 0, 1], [0, y_cart, 1], [x_cart, 0, 1]])
    res = transformation_matrix @ v.T
    print(res)
    res_norm = res / res[2]
    print(res_norm)
    max_x = np.max(res_norm[0])
    min_x = np.min(res_norm[0])
    min_x = min(min_x, 0)
    x0 = -min_x
    max_x = max(max_x, x_cart)
    max_y = np.max(res_norm[1])
    min_y = np.min(res_norm[1])
    min_y = min(min_y, 0)
    max_y = max(max_y, y_cart)
    y0 = max_y
    return (round(max_x - min_x), round(max_y - min_y), round(x0), round(y0)) # TODO round it


def project_img(img, transformation):
    inv_transf = np.linalg.inv(transformation)
    width, height, x0, y0 = transform_shape(img.shape[1], img.shape[0], transformation)
    print(img.shape)
    print(height, width)
    # height = 792
    # height, width, _ = img.shape
    input_height, input_width = img.shape[:2]
    result_img = np.zeros((height, width, 3), dtype=np.uint8)
    # result_img = np.zeros(img.shape, dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            x_cart, y_cart = x - x0, y0 - y - 1  # Cartesian coordinates
            p2_norm = np.array([x_cart, y_cart, 1])
            p1 = inv_transf @ p2_norm
            p1 = np.rint(p1 / p1[2])
            p1[1] = -p1[1] + input_height - 1
            p1 = np.array(p1 / p1[2], dtype=np.int_)
            if input_width > p1[0] >= 0 and 0 <= p1[1] < input_height:
                result_img[y, x] = img[p1[1], p1[0]]
    return result_img
#
#
# img1 = cv2.imread(f"calibration/img1.png")
# transformation_matrix = np.array([[0.7071, 0.7071, 200], [-0.7071, 0.7071, 200], [0, 0.001, 2]])
# result_img = project_img(img1, transformation_matrix)
# cv2.imshow('img', result_img)
# cv2.waitKey(0)
#
# '''
# Conclusion
# '''
#
# ## Task 3 ###
def generate_2_rows_of_A(pt_s, pt_d):
    return np.array([[pt_s[0], pt_s[1], 1, 0, 0, 0, -pt_d[0] * pt_s[0], -pt_d[0] * pt_s[1], -pt_d[0]],
                    [0, 0, 0, pt_s[0], pt_s[1], 1, -pt_d[1] * pt_s[0], -pt_d[1] * pt_s[1], -pt_d[1]]])
#
#
def find_transformation_matrix(pts_s, pts_d):
    assert len(pts_s) == len(pts_d)
    nr_of_points = len(pts_s)
    A = np.zeros((2 * nr_of_points, 9))
    for i in range(nr_of_points):
        A[2*i:2*(i+1)] = generate_2_rows_of_A(pts_s[i], pts_d[i])
    _, _, V = np.linalg.svd(A)
    eigenvector = V[-1, :]
    eigenvector_norm = eigenvector / np.linalg.norm(eigenvector)
    return eigenvector_norm.reshape(3, 3)

#
# def array_of_points_to_list(arr):
#     result = []
#     for i in range(arr.shape[0]):
#         point = []
#         for j in range(arr.shape[1] - 1):
#             point.append(arr[i, j])
#         result.append(np.array(point))
#     return result
#
#
# def test_3():
#     nr_of_tests = 20
#     np.random.seed(1234)
#     for i in range(nr_of_tests):
#         homography_matrix = np.random.random((3, 3))
#         homography_matrix /= np.linalg.norm(homography_matrix.reshape(9,1))
#         nr_of_points = np.random.randint(low=4, high=10)
#         points_s = np.random.rand(nr_of_points, 3)
#         points_s[:, 2] = 1.0
#         points_d = homography_matrix @ points_s.T
#         points_d /= points_d[2, :]
#         points_d = points_d.T
#         s = array_of_points_to_list(points_s)
#         d = array_of_points_to_list(points_d)
#         M = find_transformation_matrix(s, d)
#         if M[0,0] < 0:
#             M = -M
#         if homography_matrix[0,0] < 0:
#             homography_matrix = -homography_matrix
#         assert np.all(np.isclose(homography_matrix, M, rtol=1e-14, atol=1e-14))
#
#
# test_3()

###
'''
Conslusion
'''
###

### Task 4 ###
img1 = cv2.imread('stitching/img1.png')

img1_pts = [[295, 392], [457, 295], [771, 346], [1065, 464], [981, 380], [790, 703]]
img2_pts = [[420, 384], [569, 290], [880, 340], [1192, 463], [1102, 375], [899, 703]]
#
# homography_matrix = find_transformation_matrix(img1_pts, img2_pts)
# print(homography_matrix)
# # homography_matrix = np.array([[2,0,0],[0,2,0],[0,0,2]])
#
# # homography_matrix = np.array([[1, 0, 100], [0, 1, 100], [0, 0, 1]])
# img = project_img(img1, homography_matrix)
# cv2.imshow('img', img)
# cv2.waitKey(0)

### Task 5 ###
def find_transformed_corners(img, transformation_matrix):
    corners = np.array([[0, 0, 1], [img.shape[1], 0, 1], [img.shape[1], img.shape[0], 1], [0, img.shape[0], 1]])
    transformed_corners = transformation_matrix @ corners.T
    transformed_corners /= transformed_corners[2, :]
    result = []
    for i in range(4):
        result.append([transformed_corners[0, i], transformed_corners[1, i]])
    return result


def find_stitched_boundaries(points):
    sort_x = points.copy()
    sort_y = points.copy()
    sort_x.sort(key=lambda p: p[0])
    sort_y.sort(key=lambda p: p[1])
    xb, xe = sort_x[1][0], sort_x[-1][0]
    ye, yb = sort_y[3][1], sort_y[4][1]
    yb = -yb + 720 - 1
    ye = -ye + 720 - 1
    return int(xb), int(xe), int(yb), int(ye)


def find_overlapping_boundaries(points):
    sort_x = points.copy()
    sort_y = points.copy()
    sort_x.sort(key=lambda p: p[0])
    sort_y.sort(key=lambda p: p[1])
    xb, xe = sort_x[3][0], sort_x[5][0]
    ye, yb = sort_y[3][1], sort_y[4][1]
    yb = -yb + 720 - 1
    ye = -ye + 720 - 1
    return int(xb), int(xe), int(yb), int(ye)


def find_cutting_line(img1, img2_transformed, ops):
    diffs = np.zeros((ops[3] - ops[2], ops[1] - ops[0]))
    for i in range(ops[2], ops[3] + 1):
        for j in range(ops[0], ops[1]):
            diffs_i = i - ops[2]
            diffs_j = j - ops[0]
            diffs[diffs_i, diffs_j] = (np.sum(img1[i,j]) - np.sum(img2_transformed[i,j]))**2 # TODO change color scale and problem with i, j

    costs = np.zeros((ops[3] - ops[2], ops[1] - ops[0]))
    costs[0, :] = diffs[0, :]
    for i in range(ops[2] + 1, ops[3] + 1):
        for j in range(ops[0], ops[1]+1):
            diffs_i = i - ops[2]
            diffs_j = j - ops[0]
            prev_left = diffs[diffs_i - 1, diffs_j - 1] if diffs_j != ops[0] else float('inf')
            prev_up = diffs[diffs_i - 1, diffs_j]
            prev_right = diffs[diffs_i - 1, diffs_j + 1] if diffs_j != ops[1] else float('inf')
            costs[i, j] = diffs[diffs_i, diffs_j] + min(prev_left, prev_up, prev_right)

    result = []
    for i in range(ops[2] + 1, ops[3] + 1):
        min = float('inf')
        min_index = -1
        for j in range(ops[0], ops[1]+1):
            diffs_i = i - ops[2]
            diffs_j = j - ops[0]
            if costs[diffs_i, diffs_j] < min:
                min = costs[diffs_i, diffs_j]
                min_index = j
            result.append(min_index)

    return result


def stitch(img1, img2_transformed, bps, ops, line):
    result_img = np.zeros((bps[1] - bps[0], bps[3] - bps[2]))
    ops[2] = -ops[2] + 720 - 1
    ops[3] = -ops[3] + 720 - 1
    bps[2] = -bps[2] + 720 - 1
    bps[3] = -bps[3] + 720 - 1
    result_img[bps[0]:ops[0], bps[2]:bps[3]] = img1[bps[0]:ops[0], bps[2]:bps[3]]
    result_img[ops[1] + 1, bps[1], bps[2]:bps[3]] = img2_transformed[ops[1] + 1, bps[1], bps[2]:bps[3]]
    for i in range(ops[2] + 1, ops[3] + 1):
        for j in range(ops[0], ops[1] + 1):
            if j <= line[i]:
                result_img[i, j] = img1[i, j]
            else:
                result_img[i, j] = img2_transformed[i, j]
    return  result_img


def stitch_images_manual(img1, img2, pts1, pts2):
    homography_matrix = find_transformation_matrix(pts1, pts2)
    img2_transformed = project_img(img2, homography_matrix)
    cv2.imshow('img', img2_transformed)
    cv2.waitKey(0)
    p1, p2, p3, p4 = find_transformed_corners(img2, homography_matrix)
    p5, p6, p7, p8 = find_transformed_corners(img1, np.eye(3))
    points = [p1, p2, p3, p4, p5, p6, p7, p8]
    b1, b2, b3, b4 = find_stitched_boundaries(points)
    bps = [b1, b2, b3, b4]
    o1, o2, o3, o4 = find_overlapping_boundaries(points)
    ops = [o1, o2, o3, o4]
    line = find_cutting_line(img1, img2_transformed, ops)
    result_img = stitch(img1, img2_transformed, bps, ops, line)
    return result_img


img2 = cv2.imread('stitching/img2.png')
img = stitch_images_manual(img1, img2, img1_pts, img2_pts)
cv2.imshow('img', img)
cv2.waitKey(0)