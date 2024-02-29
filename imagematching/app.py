from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
import json


UPLOAD_FOLDER = 'C:\\Users\\malli\\Downloads\\ADS_TEAM_A_09-master\\ADS_TEAM_A_09-master\\imagematching\\upfiles'
VIDEO_UPLOAD_FOLDER = 'c:\\imagematching\\vidfiles'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


app.secret_key = 'asS5Dd21sad21asDETzH85Sd6'


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/services')
def services():
    return render_template("services.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/verify')
def verify():
    return render_template("verify.html")

@app.route('/verifyvideo')
def verifyvideo():
    return render_template("verifyvideo.html")


@app.route('/collect')
def collect():
    return render_template("form.html")


@app.route('/contact_us')
def contact_us():
    return render_template("contact_us.html")


@app.route('/submit_form', methods=['POST'])
def submit_form():

    import sqlite3
    from sqlite3 import Error
    import cv2
    import numpy as np
    import sys
    import os
    from PIL import Image
    from werkzeug.utils import secure_filename

    def create_connection(db_file):
        """ create a database connection to a SQLite database """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            print(sqlite3.version)

        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    def create_connection(db_file):
        """ create a database connection to the SQLite database
            specified by db_file
        :param db_file: database file
        :return: Connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)

        return conn

    def create_table(conn, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)

    def main():
        database = "C:\\Users\\malli\\Downloads\\ADS_TEAM_A_09-master\\ADS_TEAM_A_09-master\\db\\CriminalDetails.db"

        sql_create_projects_table = """ CREATE TABLE IF NOT EXISTS People (
                                            ID text PRIMARY KEY,
                                            Name text NOT NULL,
                                            Age text NOT NULL,
                                            Gender text NOT NULL,
                                            CN VARCHAR(10) NOT NULL,
                                            Address text NOT NULL,
                                            CR text NOT NULL
                                        ); """

        conn = create_connection(database)

        if conn is not None:

            create_table(conn, sql_create_projects_table)

        else:
            print("Error! cannot create the database connection.")

    if __name__ == '__main__':
        create_connection("C:\\Users\\malli\\Downloads\\ADS_TEAM_A_09-master\\ADS_TEAM_A_09-master\\db\\CriminalDetails.db")
        main()

    faceDetect = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imagePath = sys.argv[-1]

    def insertOrUpdate(Id, Name, Age, Gen, CN, Address, Cr):
        conn = sqlite3.connect("C:\\Users\\malli\\Downloads\\ADS_TEAM_A_09-master\\ADS_TEAM_A_09-master\\db\\CriminalDetails.db")
        cup = conn.cursor()
        cmd = "SELECT * FROM People WHERE ID="+str(Id)
        cursor = cup.execute(cmd)
        isRecordExist = 0
        for row in cursor:
            isRecordExist = 1
        if (isRecordExist == 1):
            cmd = "UPDATE People SET Name="+str(Name)+"WHERE ID="+str(Id)
            cmd2 = "UPDATE People SET Age="+str(Age)+"WHERE ID="+str(Id)
            cmd3 = "UPDATE People SET Gender="+str(Gen)+"WHERE ID="+str(Id)
            cmd4 = "UPDATE People SET CN="+str(CN)+"WHERE ID="+str(Id)
            cmd5 = "UPDATE People SET Address=" + \
                str(Address)+"WHERE ID="+str(Id)
            cmd6 = "UPDATE People SET CR="+str(Cr)+"WHERE ID="+str(Id)
            conn.execute(cmd)
        else:
            params = (Id, Name, Age, Gen, CN, Address, Cr)
            cmd = "INSERT INTO People(ID,Name,Age,Gender,CN,Address,CR) Values(?, ?, ?, ?, ?, ?, ?)"
            cmd2 = ""
            cmd3 = ""
            cmd4 = ""
            cmd5 = ""
            cmd6 = ""
            conn.execute(cmd, params)

        conn.execute(cmd2)
        conn.execute(cmd3)
        conn.execute(cmd4)
        conn.execute(cmd5)
        conn.execute(cmd6)
        conn.commit()
        conn.close()

    Id = request.form['ID']
    name = request.form['name']
    age = request.form['age']
    CN = request.form['cno']
    gen = request.form['gender']

    Address = request.form['add']
    cr = request.form['crime']

    if request.method == 'POST':

        uploaded_img = request.files['path']

        img_filename = secure_filename(uploaded_img.filename)

        uploaded_img.save(os.path.join(
            app.config['UPLOAD_FOLDER'], img_filename))

        a = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)

    print("Form submitted successfully!")

    insertOrUpdate(Id, name, age, gen, CN, Address, cr)

    sampleNum = 0
    while (True):
        image = cv2.imread(a)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum = sampleNum+1
            cv2.imwrite("dataSet/User."+str(Id)+"." +
                        str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.waitKey(100)
        cv2.imshow("Face", image)
        cv2.waitKey(1)
        if (sampleNum > 50):
            break

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'C:\\Users\\malli\\Downloads\\ADS_TEAM_A_09-master\\ADS_TEAM_A_09-master\\imagematching\\dataSet'

    def getImagesWithID(path):
        imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
        faces = []
        IDs = []
        for imagepath in imagepaths:
            if ".DS_Store" in imagepath:
                print("Junk!")
            else:
                faceImg = Image.open(imagepath).convert('L')
                faceNp = np.array(faceImg, 'uint8')
                ID = int(os.path.split(imagepath)[-1].split('.')[1])
                faces.append(faceNp)
                IDs.append(ID)
                cv2.imshow("training", faceNp)
                cv2.waitKey(10)
        return np.array(IDs), faces

    IDs, faces = getImagesWithID(path)
    recognizer.train(faces, IDs)
    recognizer.save('recognizer/trainningData.yml')
    cv2.destroyAllWindows()
    return "form submitted successfully"


@app.route('/search_img', methods=["GET", "POST"])
def search_img():
    import cv2
    import numpy as np
    import sqlite3
    import sys
    from flask import send_file
    import os

    flag = 0
    imagePath = sys.argv[-1]

    faceDetect = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if request.method == 'POST':

        uploaded_img = request.files['path']

        img_filename = secure_filename(uploaded_img.filename)

        uploaded_img.save(os.path.join(
            app.config['UPLOAD_FOLDER'], img_filename))

        a = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        # print("idhu dhan path",a)

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read("recognizer/trainningData.yml")

    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 1
    fontcolor = (0, 255, 0)

    def getProfile(id):
        conn = sqlite3.connect("C:\\Users\\malli\\Downloads\\ADS_TEAM_A_09-master\\ADS_TEAM_A_09-master\\db\\CriminalDetails.db")
        cmd = "SELECT * FROM People WHERE ID="+str(id)
        cursor = conn.execute(cmd)
        global profile
        profile = None
        for row in cursor:
            profile = row
        conn.close()
        return profile

    while (True):
        image = cv2.imread(a)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            id, conf = rec.predict(gray[y:y+h, x:x+w])

            profile = getProfile(id)
            if (profile != None):

                cv2.putText(
                    image, "Name : "+str(profile[1]), (x, y+h+20), fontface, fontscale, fontcolor)
                cv2.putText(
                    image, "Age : "+str(profile[2]), (x, y+h+45), fontface, fontscale, fontcolor)
                cv2.putText(
                    image, "Gender : "+str(profile[3]), (x, y+h+70), fontface, fontscale, fontcolor)
                cv2.putText(image, "Contact no. : " +
                            str(profile[4]), (x, y+h+95), fontface, fontscale, fontcolor)
                cv2.putText(
                    image, "Address  : "+str(profile[5]), (x, y+h+120), fontface, fontscale, fontcolor)
                cv2.putText(image, "Criminal Records : " +
                            str(profile[6]), (x, y+h+145), fontface, fontscale, fontcolor)
                if (len(str(profile[6])) == 0):

                    status = 'civillan'
                else:
                    status = str(profile[6])

            else:
                flag = 1
                cv2.putText(image, "Unknown", (x, y+h+20),
                            fontface, fontscale, fontcolor)

        cv2.namedWindow('Face', cv2.WINDOW_NORMAL)

        cv2.imshow("Face", image)
        if (cv2.waitKey(1) == ord('q')):
            break

    cv2.destroyAllWindows()

    if (flag == 0):
        return "<h1>NAME:"+str(profile[1])+"</h1><br>"+"<h1>ID:"+str(profile[0])+"</h1><br>"+"<h1>GENDER:"+str(profile[3])+"</h1><br>"+"<h1>CONT.NO:"+str(profile[4])+"</h1><br>"+"<h1>ADDRESS:"+str(profile[5])+"</h1><br>"+"<h1>"+status+"</h1><br>"
    else:

        return "<h1 align:center>"+"NO MATCH FOUND!"+"</h1><br>"+"<h2>"+"We  are  working  to  improve  our  algorithm  to  match  as  closely  as possible!!!"+"</h2><br>"


# @app.route('/search_vid', methods=["GET", "POST"])
# def search_vid():
#     import cv2
#     import numpy as np
#     import sqlite3
#     import sys
#     from flask import send_file
#     import os

#     flag = 0
#     imagePath = sys.argv[-1]

#     faceDetect = cv2.CascadeClassifier(
#         cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     if request.method == 'POST':

#         uploaded_img = request.files['path']

#         img_filename = secure_filename(uploaded_img.filename)

#         uploaded_img.save(os.path.join(
#             app.config['VIDEO_UPLOAD_FOLDER'], img_filename))

#         a = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], img_filename)
    
    # import face_recognition as fr
    # import numpy as np
    # import cv2
    # import os
    # from os import listdir

    # cap = cv2.VideoCapture(a)
    # folder_dir = "C:\\imagematching\\upfiles"

    # for images in os.listdir(folder_dir):
    #   fullpath = folder_dir+'\\'+images
    #   count = 0
    #   input_image = fr.load_image_file(fullpath)
    #   input_face_encoding = fr.face_encodings(input_image)[0]
    #   cap.set(cv2.CAP_PROP_POS_MSEC,(count*2000))
    #   while True:
    #       ret, frame = cap.read()
    #       if not ret:
    #           break
    #       rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #       face_locations = fr.face_locations(rgb_frame)

    #       # For each detected face
    #       for face_location in face_locations:
    #           top, right, bottom, left = face_location

    #           face_encoding = fr.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]

    #           similarity = fr.face_distance([input_face_encoding], face_encoding)

    #           if similarity < 0.5:
    #               print("Match found!")

    #           cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    #       cv2.imshow(frame)

    #       # Break the loop if 'q' key is pressed
    #       if cv2.waitKey(1) == ord('q'):
    #           break


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
