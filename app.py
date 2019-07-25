import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from image_processing_lib import image_processing

PORT = 8080

app = Flask(__name__)
CORS(app)
@app.route("/receipt/process/result", methods=['GET'])
def get_receipt_process_result():
    imageUrl = request.args.get('imageUrl')

    return image_processing().process_img(imageUrl)


@app.route("/", methods=['GET'])
def default():
    return "<h1> Welcome to receipt image processing server <h1>"


if __name__ == "__main__":
    app.run(host='localhost', port=PORT)