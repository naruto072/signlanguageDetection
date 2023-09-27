from detect import detect
from flask import Flask, json, request, jsonify
import os,json,pickle
import urllib.request
from werkzeug.utils import secure_filename


app = Flask(__name__)

app.secret_key = "caircocoders-ednalan"

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])



@app.route('/api/upload', methods=['POST','GET'])
def upload_image():
	d = pickle.loads(json.loads(request.data).encode('latin-1'))
	res = detect(d)
	resp = jsonify({
			'prediction' : res,
			'status' : True,
			'message' : 'Images successfully uploaded'})
	resp.status_code = 200
	return resp
	
	
if __name__ == '__main__':
    app.run(debug=True)
