import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from flask import *

#defining framework
app=Flask(__name__)

#creating web address for html pages
@app.route('/')
def index():
    return render_template("index.html")

#about the project
@app.route('/about')
def about():
    return render_template("about.html")

#loading dataset
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

#splitting the dataset (preprocess) before modelling
@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df,data
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        df=pd.read_csv(r'ThoraricSurgery.csv')
        df.head()
        df.drop('id',axis=1,inplace=True)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['DGN']= le.fit_transform(df['DGN'])
        df['PRE6']= le.fit_transform(df['PRE6'])
        df['PRE7']= le.fit_transform(df['PRE7'])
        df['PRE8']= le.fit_transform(df['PRE8'])
        df['PRE9']= le.fit_transform(df['PRE9'])
        df['PRE10']= le.fit_transform(df['PRE10'])
        df['PRE11']= le.fit_transform(df['PRE11'])
        df['PRE14']= le.fit_transform(df['PRE14'])
        df['PRE17']= le.fit_transform(df['PRE17'])
        df['PRE19']= le.fit_transform(df['PRE19'])
        df['PRE25']= le.fit_transform(df['PRE25'])
        df['PRE30']= le.fit_transform(df['PRE30'])
        df['PRE32']= le.fit_transform(df['PRE32'])
        df['Risk1Yr']= le.fit_transform(df['Risk1Yr'])
        #applying PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=16, svd_solver='full')
        pca.fit(df)
        x_pca= pca.transform(df)
        x_pca.shape
        x= pd.DataFrame(x_pca, columns=['DGN', 'PRE4', 'PRE5', 'PRE6', 'PRE7', 'PRE8', 'PRE9', 'PRE10','PRE11', 'PRE14', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'AGE'])
        #spliiting dataset for traing and testing
        x.head()
        y= df['Risk1Yr']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

#applying ML algorithms
@app.route('/model', methods=["POST","GET"])
def model():
    if request.method=="POST":
        global model
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg="Choose an algorithm")
        elif s==1:
            rf = RandomForestClassifier()
            rf = rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            ac_rf=accuracy_score(y_test,y_pred)
            ac_rf=ac_rf*100
            msg="The accuracy obtained by RandomForestClassifier is "+str(ac_rf) + str('%')
            return render_template("model.html",msg=msg)
        elif s==2:
            dt = DecisionTreeClassifier()
            dt = dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            ac_dt=accuracy_score(y_test,y_pred)
            ac_dt=ac_dt*100
            msg="The accuracy obtained by Decision Tree Classifier is "+str(ac_dt) +str('%')
            return render_template("model.html",msg=msg)
        elif s==3:
            from xgboost import XGBClassifier
            xgb = XGBClassifier()
            xgb = xgb.fit(x_train,y_train)
            y_pred = xgb.predict(x_test)
            acc_xgb=accuracy_score(y_test,y_pred)
            acc_xgb=acc_xgb*100
            msg="The accuracy obtained by XGBoost Classifier is "+str(acc_xgb) +str('%')
            return render_template("model.html",msg=msg)
        elif s==4:
            from mlxtend.classifier import StackingClassifier
            model1 =RandomForestClassifier(random_state=58)
            model3 = DecisionTreeClassifier(random_state=5)

            gnb = RandomForestClassifier()
            clf_stack = StackingClassifier(classifiers=[model1,model3], meta_classifier=gnb, use_probas=True,
                                                    use_features_in_secondary=True)
            model_stack = clf_stack.fit(x_train, y_train)
            pred_stack = model_stack.predict(x_test)
            acc_stack = accuracy_score(y_test, pred_stack)
            acc_stack=acc_stack*100
            msg="The accuracy obtained by Hybrid Model Classifier is "+str(acc_stack) +str('%')
            return render_template("model.html",msg=msg)
        elif s==5:
            from sklearn.ensemble import AdaBoostClassifier
            adb = AdaBoostClassifier()
            adb = adb.fit(x_train,y_train)
            y_pred = adb.predict(x_test)
            acc_adb=accuracy_score(y_test,y_pred)
            acc_adb=acc_adb*100
            msg="The accuracy obtained by AdaBoostClassifier  is "+str(acc_adb) +str('%')
            return render_template("model.html",msg=msg)
        elif s==6:
            from sklearn.ensemble import ExtraTreesClassifier
            etc = ExtraTreesClassifier()
            etc = etc.fit(x_train,y_train)
            y_pred = etc.predict(x_test)
            acc_etc=accuracy_score(y_test,y_pred)
            acc_etc=acc_etc*100
            msg="The accuracy obtained by ExtraTreesClassifier  is "+str(acc_etc) +str('%')
            return render_template("model.html",msg=msg)

    return render_template("model.html")

#predicting outcome through  16 input variables applying Randomforest
@app.route('/prediction' , methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        f1=float(request.form['DGN'])
        f2=float(request.form['PRE4'])
        f3=float(request.form['PRE5'])
        f4=float(request.form['PRE6'])
        f5=float(request.form['PRE7'])
        f6=float(request.form['PRE8'])
        f7=float(request.form['PRE9'])
        f8=float(request.form['PRE10'])
        f9=float(request.form['PRE11'])
        f10=float(request.form['PRE14'])
        f11=float(request.form['PRE17'])
        f12=float(request.form['PRE19'])
        f13=float(request.form['PRE25'])
        f14=float(request.form['PRE30'])
        f15=float(request.form['PRE32'])
        f16=float(request.form['AGE'])

        lee=[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]

        import pickle
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier()
        model.fit(x_train,y_train)
        result=model.predict([lee])
        if result == 0:
            msg = 'The Person will Survive'
            return render_template('prediction.html', msg=msg)
        elif result == 1:
            msg = 'The Person will Die'
            return render_template('prediction.html', msg=msg)
    return render_template("prediction.html")


if __name__=="__main__":
    app.run(debug="True",host="0.0.0.0")