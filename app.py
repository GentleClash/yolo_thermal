import os
from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import numpy as np
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO('model/best.pt')  

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file:
            for f in os.listdir(UPLOAD_FOLDER):
                os.remove(os.path.join(UPLOAD_FOLDER, f))
            # Generate unique filename
            filename = str(uuid.uuid4()) + '_' + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                return process_image(filepath, filename)
            elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
                return process_video(filepath, filename)
            else:
                return render_template('index.html', error='Unsupported file type')
    
    return render_template('index.html')

def process_image(filepath, filename):
    image = cv2.imread(filepath)
    
    # Debugging print
    print(f"Processing image: {filepath}")
    print(f"Image shape: {image.shape}")
    
    # Run YOLO detection with increased verbosity
    results = model(filepath, conf=0.5, verbose=True)
    
    # Debugging print for results
    print(f"Raw results type: {type(results)}")
    print(f"Number of results: {len(results)}")
    
    if results:
        first_result = results[0]
        
        # More detailed debugging
        print(f"First result type: {type(first_result)}")
        print(f"First result dir: {dir(first_result)}")
        
        if hasattr(first_result, 'boxes'):
            print(f"Boxes: {first_result.obb}")
            print(f"Number of boxes: {len(first_result.obb)}")
            
            try:
                annotated_image = first_result.plot()
                
                detected_classes = []
                confidence_scores = []
                
                for box in first_result.obb:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    detected_classes.append(class_name)
                    confidence_scores.append(conf)
                
                output_filename = 'result_' + filename
                output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
                cv2.imwrite(output_filepath, annotated_image)
                
                return render_template('results.html', 
                                       original_file=filename, 
                                       result_file=output_filename,
                                       detected_classes=detected_classes,
                                       confidence_scores=confidence_scores,
                                       file_type='image',
                                       zip=zip)
            
            except Exception as plot_error:
                print(f"Error plotting image: {plot_error}")
                return render_template('results.html', 
                                       original_image=filename, 
                                       error=f"Error plotting image: {plot_error}",)
        else:
            print("No boxes attribute found")
            return render_template('results.html', 
                                   original_image=filename, 
                                   error="No detection boxes found")
    else:
        print("No results returned")
        return render_template('results.html', 
                               original_image=filename, 
                               error="No results returned from model")

def process_video(filepath, filename):
    results = model(filepath, stream=True)
    
    output_filename = 'result_' + filename
    output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
    
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))
    
    # Process video frames
    for r in results:
        annotated_frame = r.plot()
        out.write(annotated_frame)
    
    cap.release()
    out.release()
    
    return render_template('results.html', 
                           original_file=filename, 
                           result_file=output_filename,
                           file_type='video',
                           zip=zip)


if __name__ == '__main__':
    app.run(debug=True)