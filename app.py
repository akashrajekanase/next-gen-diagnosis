from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

@app.route('/',methods=['GET'])
def hw():
    return render_template("index.html")


@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imageFile']
    image_path = "./images/"+imagefile.filename
    imagefile.save(image_path)

    # Load a model
    model = YOLO('yolov8.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(image_path)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        result.save(filename='result.jpg')  # save to disk

    
    

    return render_template("index.html") 

if __name__== '__main__':
    app.run(port=3000,debug=True)