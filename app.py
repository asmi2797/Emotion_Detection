import os
from flask import Flask, render_template, Response, request,jsonify,redirect,url_for
import flask_resize
from flask_cors import CORS
# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import model_from_json, load_model
from werkzeug.utils import secure_filename
import numpy as np
import imutils
import cv2
import base64
from PIL import Image
from io import StringIO, BytesIO
import numpy as np
from flask_mysqldb import MySQL
import hashlib


def temp_save(image_data, file_name="temp_face.jpg"):
    nparr = np.fromstring(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.imwrite(file_name, img)


def encode_image(img):
    ret, data = cv2.imencode('.jpg', img)
    return base64.b64encode(data)


app = Flask(__name__)
CORS(app)
app.debug = True

app.config["UPLOAD_FOLDER"] = " "#path of upload folder created in static folder

# Configure db
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root123'
app.config['MYSQL_DB'] = 'flask'

mysql = MySQL(app)

emotion_list = []

haar_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5')  # load weights
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# if a video path was not supplied, grab the reference to the webcam
# camera = cv2.VideoCapture("karikku.mp4")
def recognize_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageClone = image.copy()
    # detect faces in the input image, then clone the image so that
    # we can draw on it
    faces = haar_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    emotion_list.clear()
    # ensure at least one face was found before continuing
    for (x, y, w, h) in faces:
        cv2.rectangle(imageClone, (x, y), (x + w, y + h), (255, 0, 0), 2)
        detected_face = imageClone[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

        img_pixels = img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        predictions = model.predict(img_pixels)  # store probabilities of 7 expressions
        # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])
        emotion = emotions[max_index]
        acc = predictions[0][max_index]
        text = "{}: {:.2f}%".format(emotion, acc * 100)
        emotion_list.append(text)
        imageClone = cv2.putText(imageClone, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 255, 0), 3)
    return imageClone, len(faces),emotion_list

def verify(hash):
    # img_hash=str(hash)
    print(hash)
    cursor = mysql.connection.cursor()

    sql_fetch_blob_query = """SELECT * from image where hash = %s"""

    cursor.execute(sql_fetch_blob_query, (hash,))
    row = cursor.fetchone()
    if row is None:
        print('No image found in database')
        photo = 'temp_face.jpg'
        flag = 1
        face = 0
        stringList = " "
        # image hash is not in database
        # image is already present in database
    else:
        print('Image found in database')
        photo = row[3]
        face = row[4]
        stringList = row[5]
        flag = 0
        write_file(photo,'detected.jpg')
    return flag, face, stringList


@app.route('/process_image', methods=['POST'])
def process_image():
    if not request.json or 'msg' not in request.json:
        return 'Server Error!', 500

    # Step 1-Receiving image data from html in base64
    data = request.get_json()
    name = data['name']
    image_data = data['image_data']  # .strip('data:image/jpeg;base64,')
    image_data = image_data[23:]

    # Step 2- Saving data in temp-face so the image in is proper jpg
    img = temp_save(image_data, "temp_face.jpg")

    # Step 3-getting imagehash for temp-face
    md5hash = hashlib.md5(Image.open('temp_face.jpg').tobytes())
    hash = md5hash.hexdigest()
    # Step 4-starting mysql and verifying with database
    flag, face, emotion_list = verify(hash)

    # Step 5 -Verified using database and processing further
    if flag == 1:
        print('Loading the emotion detection')
        img_out, face, emotion_list = recognize_emotion(cv2.imread('temp_face.jpg'))
        # encoding detected image in base64
        image_data = encode_image(img_out)
        # saving detected_image in detected.jpg
        detected_image = temp_save(image_data, "detected.jpg")
        # compressing the original image
        original = Image.open('temp_face.jpg')
        original_compress = original.resize((400, 400))
        original_compress.save(os.path.join(app.config["UPLOAD_FOLDER"], name))
        # convert original_compress and detected_image to binary data to save in blob
        with open('temp_face.jpg', 'rb') as file:
            org_image = file.read()
        with open('detected.jpg', 'rb') as file:
            det_image = file.read()
        if face>0:
            print("Face present in image")
            print("Face detected :",face)
            stringList = ', '.join([str(item) for item in emotion_list])
            print("Emotions are :",stringList)
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO image(hash,name,detected,faces,emotions) VALUES(%s,%s,%s,%s,%s)",
                    (str(hash), name, det_image, str(face), stringList))
            mysql.connection.commit()
            cur.close()
        else:
            stringList = " "
            print("No face detected")
            print("No emotions")
    else:
        stringList=emotion_list
        print('inside process else ')
        image = img_to_array(cv2.imread('detected.jpg'))
        image_data = encode_image(image)


    #print(*stremotion_list, sep=", ")
    result = {'image_data': image_data.decode("utf-8"),'face':face,'emotions':stringList, 'msg': 'Operation Completed'}
    return result, 200
    #return jsonify({'image_data': image_data,'face':face,'emotions':emotion_list})


def write_file(data, filename):
    with open(filename, 'wb') as f:
        f.write(data)

classifier = load_model('Emotion_little_vgg.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']


@app.route("/live", methods=['POST'])
def live_detect():
    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            # rect,face,image = face_detector(frame)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return render_template("index.html")


@app.route("/contactus", methods=["GET", "POST"])
def contact_us():

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["mobile"]
        comments = request.form["comments"]
        # You could also use
        password = request.form.get("password")

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO query(name,emailid,phone,comments) VALUES(%s,%s,%s,%s)",
                    (name, email, str(phone), comments))
        mysql.connection.commit()
        cur.close()

    return render_template("index.html")

@app.route('/adminlogin', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            # Redirect to home page
            #return render_template("Admin_Login.html")
            return redirect(url_for('comments'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!!! Go back and try again'
            return render_template('Admin.html', msg=msg)


@app.route('/comments')
def comments():
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM query")
    data = cursor.fetchall()
    if data is None:
        msg="No comments yet"
    else:
        msg=""
    return render_template("Admin_Login.html",data=data,msg=msg)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/emotionD.html')
def static_emotion():
    return render_template("emotionD.html")

@app.route('/video.html')
def video_emotion():
    return render_template("video.html")


@app.route('/Lemotion.html')
def live_emotion():
    return render_template("Lemotion.html")


@app.route('/About.html')
def About_us():
    return render_template("About.html")


@app.route('/service.html')
def service():
    return render_template("service.html")


@app.route('/user_info.html')
def user():
    return render_template("user_info.html")


@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route('/Admin.html')
def admin():
    return render_template("Admin.html")


if __name__ == '__main__':
    app.run()
