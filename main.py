import os
from flask import Flask, render_template, request, redirect, url_for, session
from flask_cors import CORS
from pathlib import Path
import base64
from PIL import Image
from io import BytesIO
import time
from tensorflow import keras
import numpy as np
from skimage.io import imread
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import cv2

new_model = keras.models.load_model('sorbonno.h5')
benjorbonno_model = keras.models.load_model('benjonbornno.h5')

t = time.localtime()
current_time = time.strftime("%d_%m_%y_%H_%M_%S", t)
pat='static/images/'+str(current_time)+'.png'

app = Flask(__name__)


# Enter your database connection details below
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'data'
CORS(app)
# Intialize MySQL
mysql = MySQL(app)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("sorbornno.html")

@app.route("/bangla")
def bangla():
    return render_template("benjorbornno.html")


@app.route('/save', methods=['POST'])
def send():
    if request.method != 'POST':
        return make_response(jsonify({'result': 'invalid method'}), 400)
    base64_png = request.form['image']
    code = base64.b64decode(base64_png.split(',')[1])  # remove header
    image_decoded = Image.open(BytesIO(code))

    image_decoded.save(pat)
	
    # cur = mysql.connection.cursor()
    # cur.execute("INSERT INTO test_image(image) VALUES(%s)",[pat])
    # mysql.connection.commit()
    # cur.close()
    
    return render_template("sorbornno.html")

@app.route("/prdic")
def prdic():
	
    
    # cur = mysql.connection.cursor()
    # resultValue = cur.execute("SELECT * FROM `test_image` order by id DESC LIMIT 1")
    # if resultValue > 0:
    #     result = cur.fetchall()
    #     patt=result[0]
    #     patt=patt[1]

    #k="static/images/790545e8a546d11ba7888e5fd9003307.png"
    img = cv2.imread(pat)
    blurred = cv2.blur(img, (3,3))
    canny = cv2.Canny(blurred, 50, 200)

    ## find the non-zero min-max coords of canny
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)

    ## crop the region
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite("static/images/cropped.png", cropped)


    

    dimension=(28, 28)

    flat_data = []

    img = imread("static/images/cropped.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, dimension)
    flat_data.append(img_resized)



    #print(flat_data)

    flat_data = np.array(flat_data)
    flat_data = flat_data.reshape(1, 28, 28, 1)
    flat_data = flat_data.astype('float32')
    flat_data /=255

    #print(len(flat_data[0])) 
    # Use the loaded model to make predictions 
    pred = new_model.predict(flat_data.reshape(1, 28, 28, 1))
    #print("Heello")
    p=pred.argmax()
    p=p+1
    pr=0
    if p==1:
    	pr=p
    elif p==2:
    	pr=10
    elif p==3:
    	pr=11
    elif p>3:
    	pr=p-2

    filename='img/'+str(pr)+'.png'

    return render_template("sorbornno.html", filename=filename)



@app.route("/Bejorbonno_prdic")
def Bejorbonno_prdic():
	
    
    # cur = mysql.connection.cursor()
    # resultValue = cur.execute("SELECT * FROM `test_image` order by id DESC LIMIT 1")
    # if resultValue > 0:
    #     result = cur.fetchall()
    #     patt=result[0]
    #     patt=patt[1]

    #k="static/images/790545e8a546d11ba7888e5fd9003307.png"
    img = cv2.imread(pat)
    blurred = cv2.blur(img, (3,3))
    canny = cv2.Canny(blurred, 50, 200)

    ## find the non-zero min-max coords of canny
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)

    ## crop the region
    cropped = img[y1:y2, x1:x2]
    cv2.imwrite("static/images/cropped.png", cropped)


    

    dimension=(28, 28)

    flat_data = []

    img = imread("static/images/cropped.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(gray, dimension)
    flat_data.append(img_resized)



    #print(flat_data)

    flat_data = np.array(flat_data)
    flat_data = flat_data.reshape(1, 28, 28, 1)
    flat_data = flat_data.astype('float32')
    flat_data /=255

    #print(len(flat_data[0])) 
    # Use the loaded model to make predictions 
    pred = benjorbonno_model.predict(flat_data.reshape(1, 28, 28, 1))
    #print("Heello")
    p=pred.argmax()
    p=p+1

    filename='img/bbonno/'+str(p)+'.png'

    return render_template("benjorbornno.html", filename=filename)


if __name__ == "__main__":

    app.run(debug=True)
