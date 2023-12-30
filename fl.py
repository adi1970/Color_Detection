from flask import Flask,Response,render_template, request,flash,redirect, url_for,jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, validators
import os
import numpy as np
import pandas as pd
import seaborn as sns
import cv2

import matplotlib.pyplot as plt
from matplotlib import colors

import cv2
from collections import Counter
from sklearn.cluster import KMeans
UPLOAD_FOLDER = 'Static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','jfif'}
SECRET_KEY = '\xfd{H\xe5<\x95\xf9\xe3\x96.5\xd1\x01O<!\xd5\xa2\xa0\x9fR"\xa1\xa8'


app= Flask(__name__)


app.config['SECRET_KEY'] = "Your_secret_string"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(500), nullable=False)

class ContactForm(FlaskForm):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    message = StringField('Message', [validators.Length(min=1, max=500)])
    submit = SubmitField('Submit')
    
# Create the database tables
with app.app_context():
    db.create_all()


class RegistrationForm(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', [validators.Length(min=4, max=25)])
    password = PasswordField('Password', [validators.DataRequired()])
    submit = SubmitField('Login')

# Moving
@app.route("/")
def home():
    return render_template('tut.html')
def rgb_detection():
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1288)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,728)

    while True:
        _, frame=cap.read()
        hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        height, width, _ =frame.shape
    
        cx=int(width/2)
        cy=int(height/2)
    
        pixel_center=hsv_frame[cy,cx]
        hue_value=pixel_center[0]
    
        if hue_value<5:
           color='RED'
        
        elif hue_value<22:
           color='ORANGE'
        
        elif hue_value < 33:
            color='YELLOW'
        
        elif hue_value < 78:
           color='GREEN'
    
        elif hue_value < 131:
            color='BLUE'
        
        elif hue_value < 170:
            color='VIOLET'
    
        else:
           color='RED'
    
        pixel_center_bgr=frame[cy,cx]
        b,g,r=int(pixel_center_bgr[0]),int(pixel_center_bgr[1]),int(pixel_center_bgr[2])
    #print(pixel_center)
        cv2.putText(frame,color,(10,70),0,2.5,(b,g,r),2)
        cv2.circle(frame,(cx,cy),5,(25,25,25),3)
        cv2.imshow('Frame',frame)
        key=cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            cap.release() 
            cv2.destroyAllWindows() 
            break
@app.route('/video_feed')
def video_feed():
    return  Response(rgb_detection(),mimetype='multipart/x-mixed-replace; boundary=frame') 

# RGB Live
@app.route("/RGB_live")
def RGB_live():
    return render_template('tut2.html')
def rgb_Live():
    # Python code for Multiple Color Detection 

# Capturing video through webcam 
    webcam = cv2.VideoCapture(0) 

# Start a while loop 
    while(1):
        _, imageFrame = webcam.read() 

    # Convert the imageFrame in 
    # BGR(RGB color space) to 
    # HSV(hue-saturation-value) 
    # color space 
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

    # Set range for red color and 
    # define mask 
        red_lower = np.array([136, 87, 111], np.uint8) 
        red_upper = np.array([180, 255, 255], np.uint8) 
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    # Set range for green color and 
    # define mask 
        green_lower = np.array([25, 52, 72], np.uint8) 
        green_upper = np.array([102, 255, 255], np.uint8) 
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

    # Set range for blue color and 
    # define mask 
        blue_lower = np.array([94, 80, 2], np.uint8) 
        blue_upper = np.array([120, 255, 255], np.uint8) 
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
    
    # Morphological Transform, Dilation 
    # for each color and bitwise_and operator 
    # between imageFrame and mask determines 
    # to detect only that particular color 
        kernal = np.ones((5, 5), "uint8") 
    
    # For red color 
        red_mask = cv2.dilate(red_mask, kernal) 
        res_red = cv2.bitwise_and(imageFrame, imageFrame,mask = red_mask) 
    
    # For green color 
        green_mask = cv2.dilate(green_mask, kernal) 
        res_green = cv2.bitwise_and(imageFrame, imageFrame,mask = green_mask) 
    
    # For blue color 
        blue_mask = cv2.dilate(blue_mask, kernal) 
        res_blue = cv2.bitwise_and(imageFrame, imageFrame,mask = blue_mask) 

    # Creating contour to track red color 
        contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 0, 255), 2) 
            
                cv2.putText(imageFrame, "Red Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255))	 

    # Creating contour to track green color 
        contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 

        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 255, 0), 2) 
            
                cv2.putText(imageFrame, "Green Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0)) 

    # Creating contour to track blue color 
        contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(255, 0, 0), 2) 
            
                cv2.putText(imageFrame, "Blue Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 0, 0)) 
            
    # Program Termination 
        cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            webcam.release() 
            cv2.destroyAllWindows() 
            break

@app.route('/video_Live')
def video_Live():
    return  Response(rgb_Live(),mimetype='multipart/x-mixed-replace; boundary=frame') 
# Specific_red
@app.route("/Specific_red")
def spered():
    return render_template('tut3.html')
def Red():
    cap=cv2.VideoCapture(0)
    while True:
        _, frame=cap.read()
        hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #red color
        low_red=np.array([161,155,84])
        high_red=np.array([179,255,255])
        red_mask=cv2.inRange(hsv_frame,low_red,high_red)
        red=cv2.bitwise_and(frame,frame,mask=red_mask)
    
    
        cv2.imshow("Frame",frame)
        cv2.imshow("Red",red)
        key=cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            cap.release() 
            cv2.destroyAllWindows() 
            break
@app.route('/video_Red')
def video_Red():
    return  Response(Red(),mimetype='multipart/x-mixed-replace; boundary=frame') 
# Specific_green
@app.route("/Specific_green")
def spegreen():
    return render_template('tut6.html')
def Green():
    cap=cv2.VideoCapture(0)
    while True:
        _, frame=cap.read()
        hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    #Green Color
        low_green=np.array([25,52,72])
        high_green=np.array([102,255,255])
        green_mask=cv2.inRange(hsv_frame,low_green,high_green)
        green=cv2.bitwise_and(frame,frame,mask=green_mask)
    
        cv2.imshow("Frame",frame)
        cv2.imshow("Green",green)
        key=cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'): 
            cap.release() 
            cv2.destroyAllWindows() 
            break
@app.route('/video_Green')
def video_Green():
    return  Response(Green(),mimetype='multipart/x-mixed-replace; boundary=frame') 

# Specific_Blue
@app.route("/Specific_Blue")
def speblue():
    return render_template('tut7.html')
def Blue():
    cap=cv2.VideoCapture(0)
    while True:
        _, frame=cap.read()
        hsv_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
        #Blue Color
        low_blue=np.array([94,80,2])
        high_blue=np.array([129,255,255])
        blue_mask=cv2.inRange(hsv_frame,low_blue,high_blue)
        blue=cv2.bitwise_and(frame,frame,mask=blue_mask)
    
        cv2.imshow("Frame",frame)
        cv2.imshow("Blue",blue)
        key=cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'): 
           cap.release() 
           cv2.destroyAllWindows() 
           break
@app.route('/video_Blue')
def video_Blue():
    return  Response(Blue(),mimetype='multipart/x-mixed-replace; boundary=frame') 

# Dominant Color Detection
@app.route("/DominantDet")
def Dominant():
    return render_template('tut1.html')
def Dominant_Color_Detection(filename):
    def create_bar(height, width, color):
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])
        return bar, (red, green, blue)

    img = cv2.imread(f"Static/{filename}")
    height,width,_= np.shape(img)
#_= np.shape(img)
# print(height, width)

    data = np.reshape(img, (height * width, 3))
    data = np.float32(data)

    number_clusters = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)
# print(centers)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bars = []
    rgb_values = []

    for index, row in enumerate(centers):
        bar, rgb = create_bar(200, 200, row)
        bars.append(bar)
        rgb_values.append(rgb)

    img_bar = np.hstack(bars)

    for index, row in enumerate(rgb_values):
        image = cv2.putText(img_bar, f'{index + 1}. RGB: {row}', (5 + 200 * index, 200 - 10),
                        font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    print(f'{index + 1}. RGB{row}')

    cv2.imshow('Image', img)
    cv2.imshow('Dominant colors', img_bar)
# cv2.imwrite('output/bar.jpg', img_bar)

    cv2.waitKey(0)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

@app.route("/color",methods=["GET","POST"])
def color():
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "error"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            Dominant_Color_Detection(filename)


    return Dominant_Color_Detection(filename)

# GIf
@app.route("/Gif")
def Gif():
    return render_template('tut4.html')
def Gif_color_Detection(filename):
    gif = cv2.VideoCapture(f"Static/{filename}")
    while True:
        ret, imageFrame = gif.read()

    # Break the loop if there are no more frames
        if not ret:
           break

    # Convert the imageFrame to the HSV color space
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for each color
        colors = {
            'red': ([136, 87, 111], [180, 255, 255]),
            'green': ([25, 52, 72], [102, 255, 255]),
            'blue': ([94, 80, 2], [120, 255, 255])
        }

    # Detect each color in the frame
        for color, (lower, upper) in colors.items():
            mask = cv2.inRange(hsvFrame, np.array(lower), np.array(upper))
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw a rectangle around the detected objec
        
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 300:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(imageFrame, color.title() + ' Colour', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    # Display the frame
        cv2.imshow('Multiple Color Detection in Real-Time', imageFrame)
    
    # Exit the program if 'q' is pressed
        if cv2.waitKey(1000) & 0xFF == ord('q'):
           break

@app.route("/GIF",methods=["GET","POST"])
def GIF():
    if request.method == "POST":
        # check if the post request has the file part
        if 'filegif' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['filegif']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "error"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            Gif_color_Detection(filename)
    

    return Gif_color_Detection(filename)

# Static image detection
@app.route("/RGBCo")
def RGBDe():
    return render_template('tut5.html')
def RGB_Color_Detection(filename):

    index = ['color', 'color_name', 'hex', 'R', 'G', 'B']

# Path of the file in the current sysytem
    df = pd.read_csv("colors.csv", names=index, header=None)
    df.head(10)

    print('Total Number of Rows:',len(df))
    print('\n')
    print(df.loc[1])

    img=cv2.imread(f"Static/{filename}")
    print(img)                # it will print an array of number, this is the way in which computer see the array of pixel.

# Resizing the image
    img=cv2.resize(img,(800,600))

    clicked=False
    r=g=b=xpos=ypos=0
    def get_color_name(R,G,B):
        minimum = 1000
        for i in range(len(df)):
            d = abs(R - int(df.loc[i,'R'])) + abs(G - int(df.loc[i,'G'])) + abs(B - int(df.loc[i,'B']))
            if d <= minimum:
                minimum = d
                cname = df.loc[i, 'color_name']
            
        return cname
    def draw_function(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            global b, g, r, xpos, ypos, clicked
            clicked = True
            xpos = x
            ypos = y
            b,g,r = img[y,x]
            b = int(b)
            g = int(g)
            r = int(r)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_function)

    while True:
        cv2.imshow('image', img)
        if clicked:
            cv2.rectangle(img, (20,20), (600,60), (b,g,r), -1)
            text = get_color_name(r,g,b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
            cv2.putText(img, text, (50,50), 2,0.8, (255,255,255),2,cv2.LINE_AA)
            if (r+g+b) >=600:
                cv2.putText(img, text, (50,50), 2,0.8, (0,0,0),2,cv2.LINE_AA)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        return render_template('white.html')
       
@app.route("/RGBS",methods=["GET","POST"])
def RGBf():
    if request.method == "POST":
        
        if 'filergb' not in request.files:
            flash('No file part')
            return "error"
        file = request.files['filergb']
        
        if file.filename == '':
            flash('No selected file')
            return "error"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            RGB_Color_Detection(filename)

    return RGB_Color_Detection(filename)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        new_user = User(username=form.username.data, password=form.password.data)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data, password=form.password.data).first()

        if user:
            return f'Welcome, {user.username}!'
        else:
            return 'Invalid login credentials'

    return render_template('login.html', form=form)

@app.route('/users')
def display_users():
    users = User.query.all()
    return render_template('users.html', users=users)

@app.route('/submit_form', methods=['GET', 'POST'])
def submit_form():
    form = ContactForm()

    if form.validate_on_submit():
        # If the form is valid, create a new Contact instance and add it to the database
        new_contact = Contact(name=form.name.data, email=form.email.data, message=form.message.data)
        db.session.add(new_contact)
        db.session.commit()

        return redirect(url_for('success'))

    return render_template('form.html', form=form)

@app.route('/success')
def success():
    return "Form submitted successfully!"

@app.route('/view_data')
def view_data():
    contacts = Contact.query.all()
    return render_template('view_data.html', contacts=contacts)

if __name__ == '__main__':
    app.run(port=8004)