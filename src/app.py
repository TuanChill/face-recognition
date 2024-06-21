# import tkinter as tk
# from tkinter import simpledialog, messagebox, ttk
# import os
# import cv2

# import subprocess

# def run_align_dataset_mtcnn(input_dir, output_dir, image_size=160, margin=32, random_order=True, gpu_memory_fraction=0.25):
#     command = [
#         'python',
#         'src/align_dataset_mtcnn2.py',
#         input_dir,
#         output_dir,
#         '--image_size', str(image_size),
#         '--margin', str(margin),
#         '--random_order' if random_order else '',
#         '--gpu_memory_fraction', str(gpu_memory_fraction)
#     ]
    
#     subprocess.run(command)

# def train_classifier(command, loading_label):
#     try:
#         if os.path.exists("Models/facemodel.pkl"):
#             os.remove("Models/facemodel.pkl")

#         loading_label.config(text="Đang huấn luyện mô hình...", foreground="blue")
#         loading_label.update()

#         subprocess.run(command, shell=True, check=True)
#         loading_label.config(text="Hoàn thành huấn luyện mô hình", foreground="green")
#     except subprocess.CalledProcessError as e:
#         loading_label.config(text="Lỗi xảy ra khi huấn luyện mô hình", foreground="red")

# def register_user():
#     user_name = simpledialog.askstring("Đăng ký người dùng", "Nhập tên người dùng:")
#     if user_name:
#         user_dir = os.path.join("DataSet/FaceData/raw", user_name)
#         if os.path.exists(user_dir):
#             messagebox.showwarning("Cảnh báo", "Người dùng đã tồn tại!")
#             return
        
#         os.makedirs(user_dir, exist_ok=True)
        
#         cam = cv2.VideoCapture(0)
#         detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         sampleNum = 0

#         messagebox.showinfo("Thông báo", "Bắt đầu chụp ảnh sinh viên, nhấn q để thoát!")

#         while True:
#             ret, img = cam.read()
#             if not ret:
#                 break

#             img = cv2.flip(img, 1)

#             centerH = img.shape[0] // 2
#             centerW = img.shape[1] // 2
#             sizeboxW = 300
#             sizeboxH = 400
#             cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
#                           (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

#             faces = detector.detectMultiScale(img, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 sampleNum += 1
#                 face_img = img[y:y + h, x:x + w]
#                 face_img_path = os.path.join(user_dir, f"{user_name}_{sampleNum}.jpg")
#                 cv2.imwrite(face_img_path, face_img)

#             cv2.imshow('frame', img)
            
#             if cv2.waitKey(50) & 0xFF == ord('q'):
#                 break
#             elif sampleNum >= 50:
#                 break
            
#         messagebox.showinfo("Thông báo", "Bắt đầu xử lý ảnh......")
#         try:
#             run_align_dataset_mtcnn('Dataset/FaceData/raw/' + user_name, 'Dataset/FaceData/processed/' + user_name, image_size=160, margin=32, random_order=True, gpu_memory_fraction=0.25)
#             messagebox.showinfo("Thông báo", "Đã xử lý xong ảnh!")
#         except:
#             messagebox.showerror("Lỗi", "Có lỗi xảy ra khi xử lý ảnh!")
         
#         cam.release()
#         cv2.destroyAllWindows()
#         messagebox.showinfo("Thông báo", "Đăng ký thành công!")

#     user_name = simpledialog.askstring("Đăng ký người dùng", "Nhập tên người dùng:")
#     if user_name:
#         user_dir = os.path.join("DataSet/FaceData/raw", user_name)
#         if os.path.exists(user_dir):
#             messagebox.showwarning("Cảnh báo", "Người dùng đã tồn tại!")
#             return
        
#         os.makedirs(user_dir, exist_ok=True)
        
#         cam = cv2.VideoCapture(0)
#         detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         sampleNum = 0

#         messagebox.showinfo("Thông báo", "Bắt đầu chụp ảnh sinh viên, nhấn q để thoát!")

#         while True:
#             ret, img = cam.read()
#             if not ret:
#                 break

#             img = cv2.flip(img, 1)

#             centerH = img.shape[0] // 2
#             centerW = img.shape[1] // 2
#             sizeboxW = 300
#             sizeboxH = 400
#             cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
#                           (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

#             faces = detector.detectMultiScale(img, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#                 sampleNum += 1
#                 face_img = img[y:y + h, x:x + w]
#                 face_img_path = os.path.join(user_dir, f"{user_name}_{sampleNum}.jpg")
#                 cv2.imwrite(face_img_path, face_img)

#             cv2.imshow('frame', img)
            
#             if cv2.waitKey(50) & 0xFF == ord('q'):
#                 break
#             elif sampleNum >= 50:
#                 break
            
#         messagebox.showinfo("Thông báo", "Bắt đầu xử lý ảnh......")
#         try:
#             run_align_dataset_mtcnn('Dataset/FaceData/raw/' + user_name, 'Dataset/FaceData/processed/' + user_name, image_size=160, margin=32, random_order=True, gpu_memory_fraction=0.25)
#             messagebox.showinfo("Thông báo", "Đã xử lý xong ảnh!")
#         except:
#             messagebox.showerror("Lỗi", "Có lỗi xảy ra khi xử lý ảnh!")
         
#         cam.release()
#         cv2.destroyAllWindows()
#         messagebox.showinfo("Thông báo", "Đăng ký thành công!")

# def recognize_user():
#     os.system("python src/face_rec_cam.py")

# def create_loading_window():
#     loading_window = tk.Toplevel(root)
#     loading_window.title("Loading")
#     loading_window.geometry("200x100")
#     loading_label = tk.Label(loading_window, text="Đang tải...", font=("Helvetica", 12), foreground="blue")
#     loading_label.pack(pady=20)
#     loading_window.transient(root)
#     loading_window.grab_set()
#     return loading_label

# def train_model():
#     loading_label = create_loading_window()
#     train_command = 'python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000'
#     train_classifier(train_command, loading_label)

# root = tk.Tk()
# root.title("Hệ thống điểm danh sinh viên")
# root.geometry("400x300")

# frame = tk.Frame(root, bg="#e6f7ff")
# frame.pack(fill='both', expand=True)

# label_title = tk.Label(frame, text="Hệ thống điểm danh sinh viên", bg="#e6f7ff", font=("Helvetica", 16, "bold"))
# label_title.grid(row=0, column=0, columnspan=2, pady=5)

# btn_register = tk.Button(frame, text="Đăng ký người dùng", command=register_user, bg="#4CAF50", fg="white", font=("Helvetica", 12))
# btn_register.grid(row=1, column=0, padx=4, pady=4, ipadx=4, ipady=5)

# btn_recognize = tk.Button(frame, text="Nhận dạng người dùng", command=recognize_user, bg="#2196F3", fg="white", font=("Helvetica", 12))
# btn_recognize.grid(row=2, column=0, padx=4, pady=5, ipadx=5, ipady=5)

# btn_train_model = tk.Button(frame, text="Huấn luyện mô hình", command=train_model, bg="#FF5733", fg="white", font=("Helvetica", 12))
# btn_train_model.grid(row=3, column=0, padx=4, pady=5, ipadx=5, ipady=5)

# root.mainloop()


import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import os
import cv2
import subprocess

def run_align_dataset_mtcnn(input_dir, output_dir, image_size=160, margin=32, random_order=True, gpu_memory_fraction=0.25):
    command = [
        'python',
        'src/align_dataset_mtcnn2.py',
        input_dir,
        output_dir,
        '--image_size', str(image_size),
        '--margin', str(margin),
        '--random_order' if random_order else '',
        '--gpu_memory_fraction', str(gpu_memory_fraction)
    ]
    
    subprocess.run(command)

def train_classifier(command, loading_label):
    try:
        if os.path.exists("Models/facemodel.pkl"):
            os.remove("Models/facemodel.pkl")

        loading_label.config(text="Đang huấn luyện mô hình...", foreground="blue")
        loading_label.update()

        subprocess.run(command, shell=True, check=True)
        loading_label.config(text="Hoàn thành huấn luyện mô hình", foreground="green")
    except subprocess.CalledProcessError as e:
        loading_label.config(text="Lỗi xảy ra khi huấn luyện mô hình", foreground="red")

def register_user():
    user_name = simpledialog.askstring("Đăng ký người dùng", "Nhập tên người dùng:")
    if user_name:
        user_dir = os.path.join("DataSet/FaceData/raw", user_name)
        if os.path.exists(user_dir):
            messagebox.showwarning("Cảnh báo", "Người dùng đã tồn tại!")
            return
        
        os.makedirs(user_dir, exist_ok=True)
        
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        sampleNum = 0

        messagebox.showinfo("Thông báo", "Bắt đầu chụp ảnh sinh viên, nhấn q để thoát!")

        while True:
            ret, img = cam.read()
            if not ret:
                break

            img = cv2.flip(img, 1)

            faces = detector.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                face_img = img[y:y + h, x:x + w]
                face_img_path = os.path.join(user_dir, f"{user_name}_{sampleNum}.jpg")
                cv2.imwrite(face_img_path, face_img)

            cv2.imshow('frame', img)
            
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            elif sampleNum >= 50:
                break
            
        messagebox.showinfo("Thông báo", "Bắt đầu xử lý ảnh......")
        try:
            run_align_dataset_mtcnn('Dataset/FaceData/raw/' + user_name, 'Dataset/FaceData/processed/' + user_name, image_size=160, margin=32, random_order=True, gpu_memory_fraction=0.25)
            messagebox.showinfo("Thông báo", "Đã xử lý xong ảnh!")
        except:
            messagebox.showerror("Lỗi", "Có lỗi xảy ra khi xử lý ảnh!")
         
        cam.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Thông báo", "Đăng ký thành công!")

def recognize_user():
    os.system("python src/face_rec_cam.py")

def create_loading_window():
    loading_window = tk.Toplevel(root)
    loading_window.title("Loading")
    loading_window.geometry("200x100")
    loading_label = tk.Label(loading_window, text="Đang tải...", font=("Helvetica", 12), foreground="blue")
    loading_label.pack(pady=20)
    loading_window.transient(root)
    loading_window.grab_set()
    return loading_label

def train_model():
    loading_label = create_loading_window()
    train_command = 'python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000'
    train_classifier(train_command, loading_label)

root = tk.Tk()
root.title("Hệ thống điểm danh sinh viên")
root.geometry("400x300")

frame = tk.Frame(root, bg="#e6f7ff")
frame.pack(fill='both', expand=True)

label_title = tk.Label(frame, text="Hệ thống điểm danh sinh viên", bg="#e6f7ff", font=("Helvetica", 16, "bold"))
label_title.grid(row=0, column=0, columnspan=2, pady=5)

btn_register = tk.Button(frame, text="Đăng ký người dùng", command=register_user, bg="#4CAF50", fg="white", font=("Helvetica", 12))
btn_register.grid(row=1, column=0, padx=4, pady=4, ipadx=4, ipady=5)

btn_recognize = tk.Button(frame, text="Nhận dạng người dùng", command=recognize_user, bg="#2196F3", fg="white", font=("Helvetica", 12))
btn_recognize.grid(row=2, column=0, padx=4, pady=5, ipadx=5, ipady=5)

btn_train_model = tk.Button(frame, text="Huấn luyện mô hình", command=train_model, bg="#FF5733", fg="white", font=("Helvetica", 12))
btn_train_model.grid(row=3, column=0, padx=4, pady=5, ipadx=5, ipady=5)

root.mainloop()
