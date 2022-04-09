
from tensorflow.keras.models import load_model
import numpy as np 
import cv2 
import streamlit as st

model = load_model('traffic_classifier.h5')
model_cv2_resize = load_model('traffic_classifier_cv2_resize.h5')
# cap = cv2.VideoCapture(0)

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }


from PIL import Image
import streamlit as st
st.title('Traffic Sign Recognition')
def load_image(image):
	img = Image.open(image)
	return img.resize((192,192))

def predict(image):
	img = Image.open(image)
	img = img.resize((30,30))
	img_resize = np.array(img)
	final_img = np.expand_dims(img_resize,axis=0)
	print(final_img.shape)
	y_hat = model.predict(final_img)
	sign_recognited = int(np.argmax(y_hat,axis=1)+1)
	return str(classes[sign_recognited])

st.subheader("Image")

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg","jfif"],accept_multiple_files=True)
if image_file :
	row = st.columns(len(image_file))
	for image,column in zip(image_file,row):
		with column:
			st.image(load_image(image),use_column_width='auto')
			st.write(predict(image))

st.subheader("Video")
video_btn = st.button('VideoCapture')
if video_btn:
	cap = cv2.VideoCapture(0)
	while True:
		ret,frame = cap.read()

		if not ret:
			break
		frame_resize = cv2.resize(frame,(30,30))
		frame_resize = np.array(frame_resize)
		final_frame = np.expand_dims(frame_resize,axis=0)
		y_hat = model_cv2_resize.predict(final_frame)
		sign_recognited = int(np.argmax(y_hat,axis=1)+1)
		print(str(classes[sign_recognited]))
		frame = cv2.resize(frame,(500,300),interpolation = cv2.INTER_CUBIC)
		cv2.putText(frame,str(classes[sign_recognited]) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.imshow('Traffic Sign Recognition',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()