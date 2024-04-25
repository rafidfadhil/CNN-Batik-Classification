from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
app = Flask (__name__)

dic = {0 : 'Kawung',  
       1 : 'Mega_Mendung',
       2 : 'Parang', 
       3 : 'Truntum'}

model = load_model('my_model_expection.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	p = model.predict(i)
	predicted_class = np.argmax(p)
	return dic[predicted_class]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("classification.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("classification.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)