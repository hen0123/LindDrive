from flask import Flask, render_template, request #간단히 플라스크 서버를 만든다
from werkzeug.utils import secure_filename
import urllib.request
from program import posture_video
import torch
import os
import json


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('fail.html')

@app.route('/result', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST': # request.body
        video = request.form['video']
        print(video)
        path = "C:/eGovFrame-4.0.0/workspace.edu/.metadata/.plugins/org.eclipse.wst.server.core/tmp0/wtpwebapps/LDSpring/resources/upload"
        data = posture_video(video,path)
        print(data)
        return json.dumps(data, ensure_ascii=False, indent=4)
    
    else:
        return json.dumps("fail", ensure_ascii=False, indent=4)

if __name__ == '__main__':
    app.run(debug=False, host="127.0.0.1", port=5002)
