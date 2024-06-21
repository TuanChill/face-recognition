import sys
import io
import os
import argparse
import random
from time import sleep
import numpy as np
import tensorflow as tf
import facenet
import align.detect_face

# Đặt mã hóa mặc định là UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main(args):
    # Tạm dừng một khoảng thời gian ngẫu nhiên để tránh xung đột khi chạy nhiều tiến trình
    sleep(random.random())
    
    # Thiết lập thư mục đầu ra
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Tạo thư mục đầu ra nếu chưa tồn tại
    
    # Lưu thông tin phiên bản git trong thư mục log
    src_path, _ = os.path.split(os.path.realpath(__file__))
    
    # Lấy danh sách các ảnh từ thư mục đầu vào
    image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    
    print('Tạo mạng và tải các tham số')
    
    # Tạo đồ thị TensorFlow mặc định
    with tf.Graph().as_default():
        # Tạo một phiên TensorFlow
        sess = tf.compat.v1.Session()
        with sess.as_default():
            # Tạo mạng MTCNN để phát hiện khuôn mặt
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # Kích thước tối thiểu của khuôn mặt
    threshold = [0.6, 0.7, 0.7]  # Ngưỡng cho ba bước phát hiện
    factor = 0.709 # Hệ số tỷ lệ cho ảnh
    
    # Thêm một khóa ngẫu nhiên vào tên tệp để cho phép căn chỉnh bằng nhiều tiến trình
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    
    # Mở tệp để lưu tọa độ bounding box (được chú thích)
    # with open(bounding_boxes_filename, "w") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    # Nếu chọn random_order, trộn ngẫu nhiên danh sách ảnh
    if args.random_order:
        random.shuffle(image_paths)
        
    for image_path in image_paths:
        nrof_images_total += 1
        filename = os.path.splitext(os.path.split(image_path)[1])[0]
        output_filename = os.path.join(output_dir, filename + '.png')
        print(image_path)
        if not os.path.exists(output_filename):
            try:
                import imageio
                img = imageio.imread(image_path)  # Đọc ảnh từ đường dẫn
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)  # In thông báo lỗi nếu có vấn đề khi đọc ảnh
            else:
                if img.ndim < 2:
                    print('Không thể căn chỉnh "%s"' % image_path)
                    # text_file.write('%s\n' % (output_filename))
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)  # Chuyển đổi ảnh xám sang RGB
                img = img[:, :, 0:3]

                bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    det_arr = []
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces > 1:
                        if args.detect_multiple_faces:
                            for i in range(nrof_faces):
                                det_arr.append(np.squeeze(det[i]))
                        else:
                            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                            img_center = img_size / 2
                            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1], 
                                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0) # Thêm trọng số cho độ cân bằng
                            det_arr.append(det[index, :])
                    else:
                        det_arr.append(np.squeeze(det))

                    for i, det in enumerate(det_arr):
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                        bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                        bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                        bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]  # Cắt ảnh theo bounding box
                        from PIL import Image
                        cropped = Image.fromarray(cropped)  # Chuyển đổi sang đối tượng Image
                        scaled = cropped.resize((args.image_size, args.image_size), Image.BILINEAR)  # Thay đổi kích thước ảnh
                        nrof_successfully_aligned += 1
                        filename_base, file_extension = os.path.splitext(output_filename)
                        if args.detect_multiple_faces:
                            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                        else:
                            output_filename_n = "{}{}".format(filename_base, file_extension)
                        imageio.imwrite(output_filename_n, scaled)  # Lưu ảnh đã căn chỉnh
                        # text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                else:
                    print('Không thể căn chỉnh "%s"' % image_path)
                    # text_file.write('%s\n' % (output_filename))
                        
    print('Tổng số ảnh: %d' % nrof_images_total)
    print('Số ảnh căn chỉnh thành công: %d' % nrof_successfully_aligned)

def parse_arguments(argv):
    # Thiết lập các tham số cho script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Thư mục chứa ảnh chưa căn chỉnh.')
    parser.add_argument('output_dir', type=str, help='Thư mục chứa thumbnail khuôn mặt đã căn chỉnh.')
    parser.add_argument('--image_size', type=int,
        help='Kích thước ảnh (chiều cao, chiều rộng) tính bằng pixel.', default=182)
    parser.add_argument('--margin', type=int,
        help='Lề cho phần cắt xung quanh khung hình bao (chiều cao, chiều rộng) tính bằng pixel.', default=44)
    parser.add_argument('--random_order', 
        help='Trộn ngẫu nhiên thứ tự ảnh để cho phép căn chỉnh bằng nhiều tiến trình.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Giới hạn trên về lượng bộ nhớ GPU sẽ được sử dụng bởi tiến trình.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Phát hiện và căn chỉnh nhiều khuôn mặt trên mỗi ảnh.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
