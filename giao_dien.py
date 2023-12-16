from tkinter import *
import cv2
import numpy as np
from PIL import ImageGrab
from keras.models import load_model
import webbrowser

# Tải mô hình đã được huấn luyện trước
model = load_model('mnist.h5')

# Thư mục chứa ảnh được vẽ
image_folder = "img/"


# Khởi tạo giao diện đồ họa tkinter
root = Tk()
root.resizable(0, 0)
root.title("Kiểm tra số viết tay")


# Khai báo biến toàn cục
lastx, lasty = None, None
image_number = 0


# Tạo canvas để vẽ
cv = Canvas(root, width=800, height=600, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=NSEW, columnspan=2)


# Xóa nét vẽ trên canvas
def clear_widget():
   global cv
cv.delete('all')


# Vẽ nét khi di chuyển chuột
def draw_lines(event):
   global lastx, lasty
x, y = event.x, event.y
cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
lastx, lasty = x, y


# Kích hoạt sự kiện vẽ khi nhấn chuột trái
def activate_event(event):
    global lastx, lasty
cv.bind('<B1-Motion>', draw_lines)
lastx, lasty = event.x, event.y


cv.bind('<Button-1>', activate_event)


def Recognize_Digit():


   global image_number
filename = f'img_{image_number}.png'
widget = cv

x = root.winfo_rootx() + widget.winfo_rootx()
y = root.winfo_rooty() + widget.winfo_rooty()
x1 = x + widget.winfo_width()
y1 = y + widget.winfo_height()


# Lưu ảnh vẽ
ImageGrab.grab().crop((x, y, x1, y1)).save(image_folder + filename)


# Đọc ảnh và chuyển đổi sang ảnh đen trắng
image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


# Tìm contour của số trong ảnh
contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


for cnt in contours:
   x, y, w, h = cv2.boundingRect(cnt)
# Vẽ hình chữ nhật xung quanh contour
cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

# Cắt số từ ảnh tương ứng với contour hiện tại
digit = th[y:y + h, x:x + w]

# Resize số về kích thước (18, 18)
resized_digit = cv2.resize(digit, (18, 18))

# Thêm đệm với 5 pixel màu đen ở mỗi bên để tạo ra ảnh kích thước (28, 28)
padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

digit = padded_digit.reshape(1, 28, 28, 1)
digit = digit / 255.0

# Dự đoán số
pred = model.predict([digit])[0]
final_pred = np.argmax(pred)

data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

# Hiển thị kết quả trên ảnh
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.2
color = (255, 0, 0)
thickness = 1
cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

cv2.imshow('Predictions', image)
cv2.waitKey(0)


# Tạo nút kiểm tra và nút xóa chữ
btn_save = Button(text='Kiểm tra', width=15, height=3, command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Xóa chữ', width=15, height=3, command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)


# Mở giao diện người dùng
root.mainloop()

