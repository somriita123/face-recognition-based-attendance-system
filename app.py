import textwrap
import streamlit as st
import os, shutil
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from datetime import date
from datetime import datetime
from streamlit_modal import Modal
import streamlit.components.v1 as components
import pandas as pd
from streamlit_extras.stylable_container import stylable_container




nimgs = 10
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        #print(df)
        names = df['Name']
        rolls = df['Roll']
        times = df['Time']
        total = len(df)
        return names, rolls, times, total
    except Exception as e:
        print('1. Failed due to >>>>>>>>>>>>>>>>>>>>', e)
        return None, None, None, None
    

def getallusers():
    all_users = os.listdir('static/faces')
    allusers = len(all_users)
    faces = []
    faces_pics = []    

    for user in all_users:
        faces.append(user)
        imagename = os.listdir(f'static/faces/{user}')
        if imagename and imagename[0]:
            faces_pics.append(f'{user}/{imagename[0]}')    

    return allusers, faces, faces_pics

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def addNewUser(newusername):
    allusers, faces, faces_pics = getallusers()
    newuserid = allusers + 1
    #newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding New User', frame)
        try:
            cv2.setWindowProperty('Adding New User', cv2.WND_PROP_TOPMOST, 1)
        except Exception as e:
            print('Failed to set porperty%s. Reason: %s' % e)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()

def displayTotalUsers():

    allusers, faces, faces_pics = getallusers()
    st.markdown("---")
    st.markdown(
        f"""<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16"><br>
        <h5>As per our records, total number of users present in our system is - <span style='color: red'>{allusers}</span>
        </h5>""" , unsafe_allow_html=True
    )

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S %p")

    #print("=============>>>>>", userid)
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    except Exception as e:
        df = None
        print('2. Failed due to >>>>>>>>>>>>>>>>>>>>', e)

    #print("=============>>>>---->", userid.isdigit())

    if userid.isdigit():
        try:
            if int(userid) not in list(df['Roll']):
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time}')
        except ValueError:
            # do something else
            print("userid is not valid.")

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def captureAttendance():
    imgBackground=cv2.imread("static/background.png")
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            try:
                faces = extract_faces(frame)
                for (x,y,w,h) in faces:
                    #(x, y, w, h) = extract_faces(frame)[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
                    face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                    identified_person = identify_face(face.reshape(1, -1))[0]
                    #add_attendance(identified_person)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
                    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
                    cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
            except Exception as e:
                st.error(e)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        # try:
        #     cv2.setWindowProperty('Attendance', cv2.WND_PROP_TOPMOST, 1)
        # except Exception as e:
        #     st.error(e)
        
        #PRESS 'C' TO CLOSE WINDOW
        if cv2.waitKey(1) == 99 or cv2.waitKey(1) == 67:
            cap.release()
            cv2.destroyAllWindows()
            train_model()
            break
        #PRESS '0' TO TAKE ATTENDANCE
        if cv2.waitKey(1) == 48 and identified_person:
            add_attendance(identified_person)
            cap.release()
            cv2.destroyAllWindows()
            train_model()
            break
    cap.release()
    cv2.destroyAllWindows()
    train_model()

def removeFaces(user):    
    if user:
        for filename in os.listdir(f'static/faces/{user}'):
            file_path = os.path.join(f'static/faces/{user}', filename)

            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        try:
            userfolder = os.path.join(f'static/faces/', user)
            if os.path.exists(userfolder) and os.path.isdir(userfolder):
                shutil.rmtree(userfolder)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (userfolder), e)

    st.rerun()

def removeAttendance(row_id):
    #print("=============>>>>>", row_id)
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    df = df.drop(df.index[row_id])
    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)
    st.rerun()

def createTextRowColumn(dataLabel="", imagePath="", actionButtonId=""):
    if dataLabel:
        st.markdown(f'<div class="row-column">{dataLabel}</div>',unsafe_allow_html=True)
        #st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)
    elif imagePath:
        st.markdown(
            f"""<img class="row-column" src="app/static/faces/{imagePath}" alt="Streamlit logo" height="50">
            """, unsafe_allow_html=True
        )
        #st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)
    elif actionButtonId:
        with stylable_container(
            "action-button-f1",
            css_styles="""
            button {
                background-color: #DDDDDD;
                border: solid 1px #AAAAAA;
                min-Height: 20px;
                padding-top: 0;                    
                padding-bottom: 0;
                margin-top: 10px !important;
                border-radius: 3px;
                p {
                    color: #000000;
                    font-size: 14px;
                }
            }""",
        ):
            _a = actionButtonId.split("-")
            f1 = st.form(key=f'f{actionButtonId}', clear_on_submit=True, border=False)
            if f1.form_submit_button(label="Remove"):
                if _a[0] == "removeUser":
                    removeFaces(_a[1])
                elif _a[0] == "removeAttendance":
                    rowId = int(_a[1])
                    removeAttendance(rowId)

def createUserProfileTable():
    allusers, faces, faces_pics = getallusers()
    st.header("All profile user lists.")
    st.markdown("---")

    if faces_pics:

        col11, col22, col33, col44 = st.columns(4)
        with col11:
            st.markdown("<h6>Serial No</h6>",unsafe_allow_html=True)
            st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)

            for index, user in enumerate(faces_pics):
                if faces_pics[index]:
                    _sNo = index + 1
                    createTextRowColumn(str(_sNo), "", "")
        
        with col22:
            st.markdown("<h6>Profile Full Name</h6>",unsafe_allow_html=True)
            st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)

            #faces_pics.sort(key=lambda x: os.path.getmtime(x))

            for index, user in enumerate(faces_pics):
                if faces_pics[index]:
                    _f = faces_pics[index].split("/")
                    _fF = _f[0].split("_")
                    createTextRowColumn(_fF[0], "", "")

        with col33:
            st.markdown("<h6>Profile Avatar</h6>",unsafe_allow_html=True)
            st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)

            for index, user in enumerate(faces_pics):
                if faces_pics[index]:
                    createTextRowColumn("", faces_pics[index], "")

        with col44:
            st.markdown("<h6>Take Action</h6>",unsafe_allow_html=True)
            st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)

            for index, user in enumerate(faces_pics):
                if faces_pics[index]:
                    _ff = faces_pics[index].split("/")
                    createTextRowColumn("", "", f"removeUser-{_ff[0]}")

    else:

        st.markdown("<div style='text-align: center'>No user present, please proceed to add one.</div>",unsafe_allow_html=True)

def createAttendanceProfileTable():
    names, rolls, times, total = extract_attendance()
    st.header("Attendance profile user lists.")
    st.markdown("---")

    if total:

        col11, col22, col33, col44 = st.columns(4)
        with col11:
            st.markdown("<h6>Profile Id</h6>",unsafe_allow_html=True)
            st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)

            for index in range(total):
                if names[index]:
                    #_sNo = index + 1
                    createTextRowColumn(str(rolls[index]), "", "")
        
        with col22:
            st.markdown("<h6>Profile Name</h6>",unsafe_allow_html=True)
            st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)

            #faces_pics.sort(key=lambda x: os.path.getmtime(x))

            for index in range(total):
                if names[index]:
                    #_f = faces_pics[index].split("/")
                    createTextRowColumn(names[index], "", "")

        with col33:
            st.markdown("<h6>Time</h6>",unsafe_allow_html=True)
            st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)

            for index in range(total):
                if names[index]:
                    createTextRowColumn(times[index], "", "")

        with col44:
            st.markdown("<h6>Take Action</h6>",unsafe_allow_html=True)
            st.markdown('<p class="border-underscore">&nbsp;</p>', unsafe_allow_html=True)

            for index in range(total):
                if names[index]:
                    #_ff = faces_pics[index].split("/")
                    createTextRowColumn("", "", f"removeAttendance-{index}")

    else:

        st.markdown("<div style='text-align: center'>No attendance yet, please proceed to take one.</div>",unsafe_allow_html=True)

def mainDisplayBlock():
    st.markdown("---")
    st.markdown(
    """<style>
    .border-underscore {
        border-bottom: solid 1px #AAAAAA;
        height: 0;
        margin: 0
    }
    .row-column {
        padding: 5px 0 0;
        line-height: 45px;
    }
    </style>""", unsafe_allow_html=True)
    hCol1, hCol2 = st.columns(2)

    with hCol1:
        st.header("Take Attendance")
        st.markdown(
            f"""<span style="color: red; font-weight: bold">To capture today's attendannce you'll need to 
            make sure user is already present in the system.</span><br>
            Thenn press the below button named, "Take Attendance".<br>
            When pop-up window appears, please follow the instructions present in the window.
            """, unsafe_allow_html=True
        )

        with st.form(key ='Form-Take-Attendance', clear_on_submit=True):
            
            takeAttendanceSubmitted = st.form_submit_button(label = "Take Attendance")
            if takeAttendanceSubmitted:
                if 'face_recognition_model.pkl' not in os.listdir('static'):
                    st.error("There is no trained model in the static folder. Please add a new face to continue.")
                captureAttendance()

    with hCol2:
       st.header(" ")    

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        createUserProfileTable()

    with col2:
       createAttendanceProfileTable()

def mainCentralContent():
    st.title("Face Recognition Based Attendance System")

    with st.form(key ='Form-Add-New-User', clear_on_submit=True):
        with st.sidebar:
            st.header("Add New People")

            description_text = """
            Try to add as many as new people to build up the attendance system more accurate & user friendly.
            """
            description = st.empty()
            description.write(description_text.format("all"))

            addUserName = st.text_input("Provide user full name", "", autocomplete=None, placeholder="Type full name here.", help='Provided full name will be displayed n the user list')
            addUserSubmitted = st.form_submit_button(label = "Proceed Now", use_container_width=True)
            if addUserSubmitted:
                if addUserName == "":
                    st.error("User full name cann't left blank.")
                else:
                    addNewUser(addUserName)

    mainDisplayBlock()

    # sourcelines, _ = inspect.getsourcelines(demo)
    # with st.expander("Source Code"):
    #     st.code(textwrap.dedent("".join(sourcelines[1:])))
    # st.markdown(f"Credit: {url}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Recognition Based Attendance System", page_icon=":chart_with_upwards_trend:", layout="wide"
    )
    mainCentralContent()
    with st.sidebar:
        # allusers, faces, faces_pics = getallusers()
        # st.markdown("---")
        displayTotalUsers()
        # st.markdown(
        #     f"""<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16"><br>
        #     <h5>As per our records, total number of users present in our system is - <span style='color: red'>{allusers}</span><br><br>
        #     If you wish to see the full list, <a href="https://twitter.com/andfanilo">click here</a>
        #     </h5>""" , unsafe_allow_html=True
        # )

# py -m streamlit run app.py --server.port 8080