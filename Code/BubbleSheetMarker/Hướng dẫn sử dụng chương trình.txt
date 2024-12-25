1. Tải cả folder BubbleSheetMarker về, giải nén nếu ở dạng .zip. 
Trong folder gồm có 2 file ứng dụng, 2 file model, 1 file ảnh mẫu, 1 file yêu cầu thư viện và HDSD

2. Mở terminal, chuyển đường dẫn đến thư mục BubbleSheetMarker

3. Chạy câu lệnh ` pip install -r requirements.txt `

4. Có 2 phiên bản ứng dụng: 

	+ bubble_marker_no_cam.py: đầu vào là ảnh chụp trước

	+ bubble_marker_cam.py: kết nối và lấy dữ liệu trực tiếp từ camera
		+ cần lấy luồng video từ camera theo tutorial này https://www.youtube.com/watch?v=UdCSiZR8xYY
		+ sau khi có được địa chỉ, thay vào dòng 594 ' capture = open_video_stream("http://192.168.124.67:8080/video") ' 


5. Có thể chạy file bằng nút Run trên các IDE hoặc dùng câu lệnh trong terminal ( ví dụ python bubble_marker_no_cam.py )

6. Lưu ý khi chạy:
	+ Nếu chạy file no_cam, cần thay đường dẫn ảnh đầu vào ở dòng 588 ( nên đủ sáng, đủ 4 góc )

	+ Nếu chạy file cam, khuyến khích setting phân giải 1920x1440 và quality 70% và chuyển chế độ Portrait (kêt quả tốt theo thử nghiệm của nhóm)
	Các cài đặt trên có thể tìm thấy trên http://192.168.124.67:8080 ( xem lại bước 4 để lấy địa chỉ mới )

7. Khi bắt đầu chạy yêu cầu nhập các đáp án, đọc kĩ hướng dẫn đã có trên terminal.

8. Chờ đợi chương trình thực thi để thu được kết quả: mã sinh viên - mã đề - điểm