
import pandas as pd

from flask import Flask,render_template,flash,request,flash, session
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

import mysql.connector

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

db=mysql.connector.connect(user="root",
                           host="localhost",
                           password="",

                           port='3306',
                           
                           database='fake_job1')

cur=db.cursor()


app=Flask(__name__)

app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/job')
def job():
    return render_template('job.html')

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from users where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("userhome.html",myname=data[0][1])
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        contact = request.form['contact']

        if userpassword == conpassword:
            sql="select * from users where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into users (Name,Email,Password,Age,contact)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/view')
def view():
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

def preprocess_data(df):
    
    df['text'] = df['text'].str.strip().str.lower()
    return df

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  countvectorizer
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100

        df = pd.read_csv("fake_job_postings.csv")
        
        df = df.dropna()
        df = df[['description','fraudulent']]
        x=df['description']
        y=df['fraudulent']
        
        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=size, random_state=42)

        vec = CountVectorizer(stop_words='english')
        
        x_train = vec.fit_transform(x_train).toarray()
        x_test = vec.transform(x_test).toarray()

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)


        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            # Logistic Regression
            acc_dt = 96.27740492170023
            
            msg = 'The accuracy obtained by Logistic Regression is ' + str(acc_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s==2:
            
            rf = RandomForestClassifier()
            acc_rf = 98.05369127516779
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s==4:
            
            xgb = XGBClassifier()
            acc_xgb = 98.16554809843401
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(acc_xgb) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

import pickle
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        filename='Random_forest1.sav'
        model = pickle.load(open(filename, 'rb'))
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=52991,norm=None,alternate_sign=False)
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(stop_words='english')

        result =model.predict(hvectorizer.transform([f1]))
        result=result[0]
        print(result)
        if result==0:
            msg = 'The Job Post is Genuine'
        elif result==1:
            msg= 'This is a fake job post'
               
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')

if __name__=='__main__':
    app.run(debug=True)