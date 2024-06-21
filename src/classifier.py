from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import io

# Set default encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main(args):
  
    # Tạo một đồ thị tính toán mới của TensorFlow
    with tf.Graph().as_default():
      
        # Khởi tạo một phiên làm việc của TensorFlow
        with tf.compat.v1.Session() as sess:
            
            # Đặt hạt giống ngẫu nhiên để tái tạo kết quả
            np.random.seed(seed=args.seed)
            
            # Sử dụng tập dữ liệu đã chia nếu được yêu cầu
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Kiểm tra rằng có ít nhất một ảnh huấn luyện cho mỗi lớp
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'Phải có ít nhất một ảnh cho mỗi lớp trong tập dữ liệu')

            # Lấy đường dẫn và nhãn của ảnh
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('So luong lop: %d' % len(dataset))  # Thay đổi dòng này
            print('So luong anh: %d' % len(paths))    # Thay đổi dòng này
            
            # Tải mô hình đã huấn luyện trước
            print('Dang tai mo hinh trich xuat dac trung')
            facenet.load_model(args.model)
            
            # Lấy các tensor đầu vào và đầu ra
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Thực hiện forward pass để tính toán các embeddings
            print('Dang tinh toan cac dac trung cho anh')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder: images, phase_train_placeholder: False }
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Huấn luyện bộ phân loại
                print('Dang huan luyen bo phan loai')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                # Tạo danh sách tên các lớp
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Lưu mô hình bộ phân loại
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Da luu mo hinh bo phan loai vao file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Phân loại ảnh
                print('Dang kiem tra bo phan loai')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Da tai mo hinh bo phan loai tu file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Do chinh xac: %.3f' % accuracy)

  
    # # Tạo một đồ thị tính toán mới của TensorFlow
    # with tf.Graph().as_default():
      
        # Khởi tạo một phiên làm việc của TensorFlow
        with tf.compat.v1.Session() as sess:
            
            # Đặt hạt giống ngẫu nhiên để tái tạo kết quả
            np.random.seed(seed=args.seed)
            
            # Sử dụng tập dữ liệu đã chia nếu được yêu cầu
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Kiểm tra rằng có ít nhất một ảnh huấn luyện cho mỗi lớp
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'Phải có ít nhất một ảnh cho mỗi lớp trong tập dữ liệu')

            # Lấy đường dẫn và nhãn của ảnh
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Số lượng lớp: %d' % len(dataset))
            print('Số lượng ảnh: %d' % len(paths))
            
            # Tải mô hình đã huấn luyện trước
            print('Đang tải mô hình trích xuất đặc trưng')
            facenet.load_model(args.model)
            
            # Lấy các tensor đầu vào và đầu ra
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Thực hiện forward pass để tính toán các embeddings
            print('Đang tính toán các đặc trưng cho ảnh')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder: images, phase_train_placeholder: False }
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Huấn luyện bộ phân loại
                print('Đang huấn luyện bộ phân loại')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                # Tạo danh sách tên các lớp
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Lưu mô hình bộ phân loại
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Đã lưu mô hình bộ phân loại vào file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Phân loại ảnh
                print('Đang kiểm tra bộ phân loại')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Đã tải mô hình bộ phân loại từ file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Độ chính xác: %.3f' % accuracy)
                
# Hàm chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Loại bỏ các lớp có ít hơn số lượng ảnh tối thiểu
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

# Hàm phân tích các đối số đầu vào
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Chỉ định nếu cần huấn luyện một bộ phân loại mới hoặc sử dụng mô hình phân loại ' + 
        'để phân loại', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Đường dẫn đến thư mục dữ liệu chứa các mảnh mặt LFW đã căn chỉnh.')
    parser.add_argument('model', type=str, 
        help='Có thể là một thư mục chứa tệp meta và tệp ckpt hoặc một tệp mô hình protobuf (.pb)')
    parser.add_argument('classifier_filename', 
        help='Tên tệp mô hình bộ phân loại dưới dạng tệp pickle (.pkl). ' + 
        'Đối với huấn luyện đây là đầu ra và đối với phân loại đây là đầu vào.')
    parser.add_argument('--use_split_dataset', 
        help='Chỉ định rằng tập dữ liệu được chỉ định bởi data_dir nên được chia thành tập huấn luyện và kiểm tra. ' +  
        'Nếu không thì có thể chỉ định một tập kiểm tra riêng bằng tùy chọn test_data_dir.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Đường dẫn đến thư mục dữ liệu kiểm tra chứa các ảnh đã căn chỉnh được sử dụng để kiểm tra.')
    parser.add_argument('--batch_size', type=int,
        help='Số lượng ảnh xử lý trong một lô.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Kích thước ảnh (chiều cao, chiều rộng) tính bằng pixel.', default=160)
    parser.add_argument('--seed', type=int,
        help='Hạt giống ngẫu nhiên.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Chỉ bao gồm các lớp có ít nhất số lượng ảnh này trong tập dữ liệu', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Sử dụng số lượng ảnh này từ mỗi lớp để huấn luyện và số còn lại để kiểm tra', default=20)
    
    return parser.parse_args(argv)

# Chạy hàm main nếu tệp được chạy trực tiếp
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
