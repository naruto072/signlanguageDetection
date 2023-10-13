#from detect import detect
from flask import Flask, json, request, jsonify
import os,json,pickle
import urllib.request



app = Flask(__name__)

app.secret_key = "caircocoders-ednalan"






@app.route('/api/upload', methods=['POST','GET'])
def upload_image():
	#d = pickle.loads(json.loads(request.data).encode('latin-1'))
	#res = detect(d)
	resp = jsonify({
			'prediction' : 'res',
			'status' : True,
			'message' : 'Images successfully uploaded'})
	resp.status_code = 200
	return resp
	
	
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
