import os
from ultralytics import YOLO
import cv2
import numpy as np


def load_model(model_path):
    return YOLO(model_path)

def open_video_stream(url):
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        print("Không thể mở luồng video")
        exit()
    return capture

def get_corner_coord(frame, model):
    '''
    Hàm chạy model để lấy toạ độ các góc
    :param frame: Ảnh đầu vào chứa phiếu chấm thi
    :param model: model sử dụng
    :return: toạ độ 4 góc của phiếu chấm thi trong ảnh
    '''
    # Chỉnh cỡ ảnh nhỏ đi để tăng tốc xử lí model
    original_height, original_width = frame.shape[:2]
    resized_frame = cv2.resize(frame, (512, 512))

    # Lấy tỉ lệ đã resize để resize lại tâm đúng với ảnh gốc
    scale_x = original_width / 512
    scale_y = original_height / 512

    # Gọi ra model, sử dụng model với ảnh cỡ 512x512
    results = model(resized_frame, imgsz=512)

    # Lấy ra kết quả các bounding box từ model
    boxes = results[0].boxes

    # Tạo mảng chứa toạ độ các góc
    corners = []
    if len(boxes) == 4:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            # Chỉnh lại toạ độ cho đúng toạ độ ảnh gốc
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            # tính tâm
            corner_x = int((x1 + x2) / 2)
            corner_y = int((y1 + y2) / 2)
            corners.append((corner_x, corner_y))

    return corners
def crop_img(image, original_corners):
    '''
    Hàm cắt bớt phần thừa trong ảnh, chỉ giữ lại phần chứa giấy chấm thi
    :param image: ảnh đầu vào cần crop
    :param original_corners: toạ độ các góc của phiếu chấm thi có trong ảnh đầu vào
    :return: ảnh mới đã được crop sát với phiếu chấm thi và toạ độ của 4 góc phiếu chấm thi trên ảnh mới
    '''

    # Tìm giá trị min và max của x và y
    min_x = min([coord[0] for coord in original_corners])
    max_x = max([coord[0] for coord in original_corners])
    min_y = min([coord[1] for coord in original_corners])
    max_y = max([coord[1] for coord in original_corners])

    # Cắt ảnh
    cropped_image = image[min_y:max_y, min_x:max_x]

    # Tính lại tọa độ của các điểm trong ảnh mới
    new_coordinates = [(coord[0] - min_x, coord[1] - min_y) for coord in original_corners]

    # Lưu ảnh đã cắt
    cv2.imwrite("cropped_image.jpg", cropped_image)

    return cropped_image, new_coordinates
def sort_coordinates(points):
    '''
    Hàm sắp xếp toọ độ theo thứ tự tl - tr - br - bl
    :param points: Toạ độ 4 góc của phiếu chấm thi không theo thứ tự
    :return: Toạ độ 4 góc của phiếu chấm thi đã theo thứ tự
    '''
    # Sắp xếp theo y trước, nếu bằng nhau thì xét x
    points = sorted(points, key=lambda p: (p[1], p[0]))

    # Hai điểm trên cùng
    top_points = points[:2]
    # Hai điểm dưới cùng
    bottom_points = points[2:]

    # Phân loại trái phải
    tl = min(top_points, key=lambda p: p[0])  # Top-Left
    tr = max(top_points, key=lambda p: p[0])  # Top-Right
    bl = min(bottom_points, key=lambda p: p[0])  # Bottom-Left
    br = max(bottom_points, key=lambda p: p[0])  # Bottom-Right

    return [tl, tr, br, bl]
def warp_perspective(image, corners, output_width=1240, output_height=1754):
    '''
    Hàm làm thẳng ảnh
    :param image: ảnh đầu vào cần làm thẳng ( chứa phiếu chấm thi )
    :param corners: toạ độ 4 góc của phiếu chấm thi trong ảnh
    :param output_width: chiều rộng ảnh đầu ra mong muốn ( mặc định là 1240 - 1/2 tỉ lệ chuẩn giấy A4)
    :param output_height: chiều dài ảnh đầu ra mong muốn ( mặc định là 1754 - 1/2 tỉ lệ chuẩn giấy a4)
    :return:
    '''

    corners = sort_coordinates(corners)  # sắp xếp các toạ độ

    # https://stackoverflow.com/questions/63954772/perspective-transform-in-opencv-python
    input = np.float32(corners)
    output = np.float32([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]])

    # compute perspective matrix
    matrix = cv2.getPerspectiveTransform(input, output)

    # do perspective transformation setting area outside input to black
    warped_img = cv2.warpPerspective(image, matrix, (output_width, output_height), cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

    # save the warped output
    cv2.imwrite("warped_final.jpg", warped_img)

    return warped_img

def histogram_equalization_and_binarization(image_path, clip_limit=2.0, tile_grid_size=(8, 8), threshold_value=120, output_path='binary_img.jpg'):
    """
    Xử lý ảnh bao gồm cân bằng histogram, áp dụng CLAHE và nhị phân hóa.

    Parameters:
        image_path (str): Đường dẫn đến ảnh gốc.
        clip_limit (float): Giá trị clipLimit cho CLAHE.
        tile_grid_size (tuple): Kích thước tileGridSize cho CLAHE.
        threshold_value (int): Ngưỡng nhị phân hóa.
        output_path (str): Đường dẫn lưu ảnh nhị phân hóa.

    Returns:
        binary_img: ảnh nhị phân hoá
    """
    # Đọc ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không thể đọc ảnh từ đường dẫn: " + image_path)

    # CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_img = clahe.apply(img)

    # Nhị phân hóa (Thresholding)
    _, binary_img = cv2.threshold(clahe_img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    # Lưu ảnh nhị phân hóa
    cv2.imwrite(output_path, binary_img)

    return binary_img

def morphological_transform(binary_image, erode_kernel_size=(17, 17), dilate_kernel_size=(5,5), iterations=1, output_erode_path='eroded_image.jpg', output_dilate_path='dilated_image.jpg'):
    """
    Thực hiện biến đổi hình thái học (erode và dilate) để xử lý ảnh nhị phân.

    Parameters:
        erode_kernel_size (tuple) kích thước kernel cho erode
        dilate_kernel_size (tuple) kích thước kernel cho dilate
        iterations (int): Số lần lặp cho erode và dilate.
        output_erode_path (str): Đường dẫn lưu ảnh sau khi erode.
        output_dilate_path (str): Đường dẫn lưu ảnh sau khi dilate.

    Returns:
        dilated_image: ảnh sau khi biến đổi hình thái chỉ còn lại các keypoint
    """

    # Tạo kernel
    erode_kernel = np.ones(erode_kernel_size, np.uint8)
    dilate_kernel = np.ones(dilate_kernel_size, np.uint8)
    # Áp dụng erode và dilate
    eroded_image = cv2.erode(binary_image, erode_kernel, iterations=iterations)
    dilated_image = cv2.dilate(eroded_image, dilate_kernel, iterations=iterations)

    # Lưu ảnh sau biến đổi
    cv2.imwrite(output_erode_path, eroded_image)
    cv2.imwrite(output_dilate_path, dilated_image)

    return dilated_image

def detect_white_point(dilated_image, blur_kernel=(5, 5), threshold_value=200):
    """
    Phát hiện các điểm trắng trên ảnh.

    Parameters:
        dilated_image (object): Ảnh đã xử lý hình thái chỉ còn lại các keypoint.
        blur_kernel (tuple): Kích thước kernel làm mờ Gaussian.
        threshold_value (int): Ngưỡng nhị phân để phát hiện chấm trắng.
        output_path (str): Đường dẫn lưu ảnh đã vẽ các chấm trắng.

    Returns:
        list: Danh sách tọa độ (x, y) của các chấm trắng.
    """
    #
    # Làm mờ ảnh để loại bỏ nhiễu nhỏ
    blurred = cv2.GaussianBlur(dilated_image, blur_kernel, 0)

    # Phát hiện các chấm trắng bằng ngưỡng nhị phân
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Tìm contours của các chấm trắng
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo danh sách lưu tọa độ các chấm
    coordinates = []

    # Lặp qua từng contour để lấy tọa độ trung tâm
    for contour in contours:
        # Tính moments để tìm trung tâm contour
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Tránh chia cho 0
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordinates.append((cX, cY))

    # Sắp xếp tọa độ theo trục y và x để tiện quan sát
    coordinates = sorted(coordinates, key=lambda x: (x[1], x[0]))

    # Trả về danh sách tọa độ
    return coordinates

def group_points_by_y(points, epsilon=30):
    """
    Nhóm các điểm theo giá trị y gần nhau và sắp xếp trong nhóm theo trục X (từ trái sang phải).

    points: danh sách các điểm (x, y)
    epsilon: sai số cho phép (đơn vị: pixel)

    Trả về danh sách các nhóm điểm đã sắp xếp.
    """
    # Sắp xếp các điểm theo tọa độ y
    points_sorted = sorted(points, key=lambda p: p[1])

    # Khởi tạo danh sách nhóm
    groups = []
    current_group = [points_sorted[0]]

    for i in range(1, len(points_sorted)):
        if abs(points_sorted[i][1] - current_group[-1][1]) <= epsilon:
            current_group.append(points_sorted[i])
        else:
            if len(current_group) >= 2:
                # Sắp xếp nhóm theo trục X
                groups.append(sorted(current_group, key=lambda p: p[0]))
            current_group = [points_sorted[i]]

    # Thêm nhóm cuối cùng nếu còn lại và đủ điều kiện
    if len(current_group) >= 2:
        groups.append(sorted(current_group, key=lambda p: p[0]))

    return groups

def remove_4_corners(groups):
    """
    Loại bỏ các điểm tl, tr, bl, br từ danh sách nhóm, giữ nguyên cấu trúc nhóm.

    groups: danh sách các nhóm điểm đã được gom nhóm.

    Trả về danh sách các nhóm còn lại sau khi loại bỏ.
    """
    # Lấy nhóm đầu tiên và nhóm cuối cùng
    top_group = groups[0]  # Nhóm 1
    bottom_group = groups[-1]  # Nhóm cuối cùng

    # Tạo danh sách nhóm còn lại
    remain_groups = []
    for i, group in enumerate(groups):
        if i == 0:  # Loại bỏ phần tử đầu và cuối trong nhóm đầu tiên
            remain_groups.append(group[1:-1])
        elif i == len(groups) - 1:  # Loại bỏ phần tử đầu và cuối trong nhóm cuối cùng
            remain_groups.append(group[1:-1])
        else:  # Giữ nguyên các nhóm khác
            remain_groups.append(group)

    return remain_groups
def get_width_from_points(p1, p2):
    return abs(p2[0] - p1[0])
def crop_image(img, points, width=None):
    """
    Cắt ảnh sử dụng perspective transform
    Input:
        - img: ảnh đầu vào
        - points: danh sách 4 điểm [tl, tr, bl, br] hoặc 2 điểm [tl, bl]
        - width: chiều rộng của ảnh output (chỉ cần khi dùng 2 điểm)
    Output:
        - ảnh đã cắt và transform
        - tọa độ góc (sau khi transform)
    """
    # Xử lý trường hợp 2 điểm - tạo thêm tr và br
    if len(points) == 2:
        tl, bl = points
        if width is None:
            raise ValueError("Cần width để cắt ảnh từ 2 điểm")
        tr = (tl[0] + width, tl[1])  # điểm tr cách tl một khoảng width
        br = (bl[0] + width, bl[1])  # điểm br cách bl một khoảng width
    else:
        tl, tr, bl, br = points

    # Tính kích thước ảnh output
    width = abs(tr[0] - tl[0])   # chiều rộng = khoảng cách từ tl đến tr
    height = abs(bl[1] - tl[1])  # chiều cao = khoảng cách từ tl đến bl

    # Tọa độ 4 góc ảnh nguồn
    source = np.float32([tl, tr, bl, br])

    # Tọa độ 4 góc ảnh đích (hình chữ nhật chuẩn)
    destination = np.float32([
        [0, 0],           # top-left
        [width, 0],       # top-right
        [0, height],      # bottom-left
        [width, height]   # bottom-right
    ])

    # Tính ma trận transform và áp dụng
    matrix = cv2.getPerspectiveTransform(source, destination)
    result = cv2.warpPerspective(img, matrix, (width, height))

    return (tl, tr, bl, br), result

def process_bubble_sheet(warped_img, points):
    results = []
    # Ảnh 1: khung mã sinh viên
    tl = points[0][0]
    tr = points[0][1]
    bl = points[1][0]
    br = points[1][1]
    ans_coords, ans_frame = crop_image(warped_img, [tl, tr, bl, br])
    width_ans = abs(tr[0] - tl[0])
    results.append(("id_frame", ans_frame, ans_coords))

    # Ảnh 2: khung mã đề
    tl = points[0][1]  # điểm tr của khung đáp án trở thành tl của khung mã đề
    bl = points[1][1]  # điểm br của khung đáp án trở thành bl của khung mã đề
    code_coords, code_frame = crop_image(warped_img, [tl, bl], width=width_ans//2)
    results.append(("code_frame", code_frame, code_coords))

    # Xử lý khung câu hỏi (3x4) - theo thứ tự cột trước, hàng sau
    for col in range(4):
        for row in range(3):
            base_row = row + 2  # Bắt đầu từ hàng thứ 3 trong points

            if col < 3:  # 3 cột đầu có đủ 4 điểm
                tl = (points[base_row][col][0], points[base_row][col][1])
                tr = (points[base_row][col+1][0], points[base_row][col+1][1])
                bl = (points[base_row+1][col][0], points[base_row+1][col][1])
                br = (points[base_row+1][col+1][0], points[base_row+1][col+1][1])
                section_coords, section = crop_image(warped_img, [tl, tr, bl, br])

            else:  # Cột cuối dùng width của cột trước
                tl = (points[base_row][col][0], points[base_row][col][1])
                bl = (points[base_row+1][col][0], points[base_row+1][col][1])
                prev_width = get_width_from_points(points[base_row][col-1], points[base_row][col])
                section_coords, section = crop_image(warped_img, [tl, bl], width=prev_width)

            results.append((f"section_col{col}_row{row}", section, section_coords))

    return results
def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    index = 1
    for name, img, coords in results:
        # Tạo tên file theo tọa độ 4 góc
        tl, tr, bl, br = coords  # Lấy tọa độ 4 góc từ coords

        # Chuyển tọa độ thành chuỗi để tạo tên file (có thể làm tròn các giá trị tọa độ nếu cần)
        filename = f"{index:02d}.({int(tl[0])},{int(tl[1])}),({int(tr[0])},{int(tr[1])}),({int(br[0])},{int(br[1])}),({int(bl[0])},{int(bl[1])}).jpg"
        output_path = os.path.join(output_dir, filename)

        # Lưu ảnh
        cv2.imwrite(output_path, img)
        index += 1
def convert_answers_to_positions(all_answers):
    max_circles = max((max(answers) if answers else 0) for answers in all_answers)
    result = [[] for _ in range(max_circles)]
    for roi_index, answers in enumerate(all_answers):
        for position in answers:
            result[position - 1].append(roi_index)
    return result
def extract_information(model, sections_path='output'):

    student_id = ''
    exam_code = ''
    # Lấy danh sách tất cả các file hình ảnh trong thư mục
    image_paths = [
        os.path.join(sections_path, file)
        for file in os.listdir(sections_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg'))  # Chỉ chọn các file ảnh
    ]

    image_paths = sorted(image_paths)
    # Danh sách lưu kết quả đáp án từ tất cả các ảnh
    all_answers_per_image = []

    count = 1

    # Duyệt qua từng ảnh trong danh sách
    for image_path in image_paths:
        # Đọc ảnh gốc
        img = cv2.imread(image_path)

        # Thay đổi kích thước ảnh gấp 2 lần
        height, width = img.shape[:2]
        img_resized = cv2.resize(img, (width * 2, height * 2))

        # Chạy inference trên ảnh đã thay đổi kích thước
        results = model(img_resized, conf=0.7, iou=0.1)

        # Lấy các thông tin dự đoán từ results
        boxes = results[0].boxes  # các bounding boxes
        confidence = boxes.conf  # xác suất (confidence)

        # Chuyển ảnh resized thành grayscale (ảnh xám)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Sử dụng Otsu để phân ngưỡng tự động cho ảnh grayscale
        _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Lọc các bounding box vượt ngưỡng
        threshold = 0.7
        filtered_boxes = [
            (boxes.xywh[i].tolist(), confidence[i].item())  # Lưu thông tin box và xác suất
            for i in range(len(boxes.xywh))
            if confidence[i].item() >= threshold
        ]

        # Sắp xếp các box theo thứ tự từ trên xuống và từ trái qua phải
        filtered_boxes_sorted = sorted(filtered_boxes,
                                       key=lambda x: (x[0][1], x[0][0]))  # Sắp xếp theo y_center và x_center

        # Danh sách lưu kết quả các hình tròn
        circles_detected = []
        # Danh sách lưu đáp án toàn cục từ tất cả các ROI
        all_answers = []

        # Duyệt qua từng box đã sắp xếp và phát hiện hình tròn
        for idx, (box, conf) in enumerate(filtered_boxes_sorted):
            x_center, y_center, w, h = box
            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

            roi = binary_img[y1:y2, x1:x2]

            # Khử nhiễu và làm sắc nét ảnh trước khi áp dụng Canny
            roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)

            # Tạo kernel sharpening
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])

            # Áp dụng sharpening cho ROI
            roi_sharpened = cv2.filter2D(roi_blurred, -1, kernel)

            # Áp dụng Canny Edge Detection trên ảnh đã làm sắc nét
            edges = cv2.Canny(roi_sharpened, threshold1=100, threshold2=200)

            # Làm dày các cạnh bằng cách sử dụng Dilate
            kernel = np.ones((3, 3), np.uint8)  # Tạo hạt nhân 3x3
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # Phát hiện hình tròn
            circles = cv2.HoughCircles(
                dilated_edges,
                cv2.HOUGH_GRADIENT,
                dp=0.5,
                minDist=50,
                param1=40,
                param2=25,
                minRadius=8,
                maxRadius=30
            )

            # Kiểm tra nếu có hình tròn được phát hiện
            if circles is not None:
                circles = np.uint16(np.around(circles))  # Làm tròn tọa độ
                # Sắp xếp hình tròn theo thứ tự x (từ trái qua phải), và nếu x giống nhau, sắp xếp theo y
                circles_sorted = sorted(circles[0, :], key=lambda x: (x[0], x[1]))

                # Danh sách lưu đáp án trong từng ROI
                answers_in_roi = []

                # In ra kết quả cho từng ROI
                for i, circle in enumerate(circles_sorted):
                    cx, cy, r = circle  # Tọa độ tâm và bán kính của hình tròn
                    circles_detected.append((cx + x1, cy + y1, r))
                    cv2.circle(img_resized, (cx + x1, cy + y1), r, (0, 255, 0), 2)  # Vẽ hình tròn màu xanh
                    cv2.circle(img_resized, (cx + x1, cy + y1), 2, (0, 0, 255), 3)  # Vẽ tâm hình tròn màu đỏ

                    # Kiểm tra pixel đen trong hình tròn
                    black_pixel_values = []
                    for angle in range(0, 360, 3):
                        for radius in range(0, r, 2):
                            x = int(cx + radius * np.cos(np.radians(angle)))
                            y = int(cy + radius * np.sin(np.radians(angle)))
                            if 0 <= x < roi.shape[1] and 0 <= y < roi.shape[0]:
                                if roi[y, x] == 0:
                                    black_pixel_values.append((x, y))

                    # Nếu số lượng pixel đen lớn hơn 1000, thêm vào đáp án
                    if len(black_pixel_values) > 1000:
                        answers_in_roi.append(i + 1)  # Thêm thứ tự hình tròn vào đáp án

                # Thêm đáp án của ROI vào danh sách toàn cục
                all_answers.append(answers_in_roi)

        # Thêm đáp án của ảnh vào danh sách toàn cục
        if (count == 1):
            id = convert_answers_to_positions(all_answers)
            # Kiểm tra nếu toàn bộ danh sách trống
            if all(not pos for pos in id):
                student_id = "không hợp lệ"
            else:
                # Chuyển mảng vị trí thành chuỗi số báo danh
                student_id = "".join(
                    str(pos[0]) for pos in id if pos)  # Lấy vị trí đầu tiên trong mỗi danh sách con nếu không rỗng
            count += 1

        elif (count == 2):
            code = convert_answers_to_positions(all_answers)
            # Kiểm tra nếu toàn bộ danh sách trống
            if all(not pos for pos in code):
                exam_code = "không hợp lệ"
            else:
                # Chuyển mảng vị trí thành chuỗi số báo danh
                exam_code = "".join(
                    str(pos[0]) for pos in code if pos)  # Lấy vị trí đầu tiên trong mỗi danh sách con nếu không rỗng
            count += 1
        else:
            all_answers_per_image.append(all_answers)

    # Làm phẳng all_answers_per_image
    final_answer = [answers for all_answers in all_answers_per_image for answers in all_answers]

    return student_id, exam_code, final_answer
def create_label_answer():
    label_answer = []
    question_number = 1
    print("Nhập đáp án nhãn cho từng câu hỏi")
    print(f"Với câu có nhiều đáp án thì các đáp án cách nhau bằng dấu ','. Ví dụ: Đáp án câu x: 2,3")
    print("Nhập 'done' để kết thúc nhập đáp án")
    while True:
        # Hiển thị lời nhắc nhập đáp án cho từng câu hỏi
        user_input = input(f"\u0110áp án câu {question_number}: ")

        # Kiểm tra nếu người dùng nhập "done" để kết thúc
        if user_input.lower() == "done":
            print("Hoàn tất việc nhập đáp án.")
            break

        try:
            # Tách các số bằng dấu phẩy và chuyển thành danh sách các số nguyên
            answers = [int(num.strip()) for num in user_input.split(",")]
            label_answer.append(answers)  # Thêm đáp án vào danh sách
            question_number += 1  # Chuyển sang câu hỏi tiếp theo
        except ValueError:
            print("\u0110ịnh dạng nhập không hợp lệ. Hãy nhập các số nguyên cách nhau bằng dấu phẩy.")

    # Trả về danh sách label_answer
    return label_answer
def compare_answers(final_answer, label_answer):

    correct_count = 0

    # Duyệt qua cả hai danh sách
    for final, label in zip(final_answer, label_answer):
        # Nếu đáp án trùng khớp hoàn toàn
        if sorted(final) == sorted(label):
            correct_count += 1

    return correct_count
def calculate_score(final_answer, label_answer):

    # Tính số câu đúng bằng cách sử dụng hàm compare_answers
    correct_count = compare_answers(final_answer, label_answer)

    # Tính tổng số câu
    total_questions = len(label_answer)

    # Tính điểm trên thang 10
    score = (correct_count / total_questions) * 10
    return score

def main():
    corner_model = load_model('corner.pt')
    line_model = load_model('line_detect.pt')

    label_answer = create_label_answer()  # cho phép người dùng nhập nhãn

    capture = open_video_stream("http://192.168.124.67:8080/video")

    frame_count = 0
    corners = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if frame_count % 10 == 0:
            corners = get_corner_coord(frame, corner_model)
            if len(corners) == 4:
                cv2.imwrite('detection_frame.jpg', frame)
                cropped_img, new_corner_coord = crop_img(frame, corners)  # cắt bỏ các vùng không chứa giấy

                warped_img = warp_perspective(cropped_img, new_corner_coord)  # làm "thẳng" giấy

                binary_image = histogram_equalization_and_binarization("warped_final.jpg")  # chuyển ảnh nhị phân

                dilated_image = morphological_transform(
                    binary_image)  # biến đổi hình thái để xác định các keypoint ( các ô vuông )

                all_keypoints = detect_white_point(dilated_image)  # trích xuất tất cả các keypoint có trong ảnh

                grouped_keypoints = group_points_by_y(all_keypoints, 30)  # phân nhóm keypoint

                section_key_point = remove_4_corners(
                    grouped_keypoints)  # Loại bỏ 4 keypoint to ở 4 góc, chỉ để lại keypoint ở vùng có tô

                sheet_process_result = process_bubble_sheet(warped_img,
                                                            section_key_point)  # tách ra các section tương ứng với mã sinh viên, mã đề thi và vùng đáp án

                save_results(sheet_process_result, 'output')  # lưu kết quả các section

                student_id, exam_code, final_answer = extract_information(line_model)  # lấy ra thông tin

                score = calculate_score(final_answer, label_answer)  # tính điểm

                print("số báo danh:", student_id)
                print("mã đề thi:", exam_code)
                print("điểm cuối cùng:", score)

                break

        cv2.imshow('YOLO Detection', frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
