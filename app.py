import os
from flask import Flask, render_template, request, redirect
from ultralytics import YOLO
import cv2
import numpy as np
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
CROPS_FOLDER = 'static/crops'  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPS_FOLDER, exist_ok=True) 

model = YOLO('model/best.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file:
            for folder in [UPLOAD_FOLDER, CROPS_FOLDER]:
                for f in os.listdir(folder):
                    os.remove(os.path.join(folder, f))
            
            filename = str(uuid.uuid4()) + '_' + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                return process_image(filepath, filename)
            elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                return process_video(filepath, filename)
            else:
                return render_template('index.html', error='Unsupported file type')
    
    return render_template('index.html')

def process_image(filepath, filename):
    results = model(filepath, conf=0.8, verbose=True)
    
    if results and len(results) > 0:
        first_result = results[0]
        
        if hasattr(first_result, 'boxes'):
            annotated_image = first_result.plot()
            
            detected_objects = []
            original_image = cv2.imread(filepath)
            
            for idx, box in enumerate(first_result.obb):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cropped = original_image[y1:y2, x1:x2]
                
                crop_filename = f'crop_{idx}_{class_name}_{filename}'
                crop_filepath = os.path.join(CROPS_FOLDER, crop_filename)
                cv2.imwrite(crop_filepath, cropped)
                
                detected_objects.append({
                    'class_name': class_name,
                    'confidence': conf,
                    'crop_filename': crop_filename
                })
            
            output_filename = 'result_' + filename
            output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
            cv2.imwrite(output_filepath, annotated_image)
            
            return render_template('results.html',
                               original_file=filename,
                               result_file=output_filename,
                               detected_objects=detected_objects,
                               file_type='image')
        
        else:
            return render_template('results.html',
                               original_file=filename,
                               error="No detection boxes found")
    else:
        return render_template('results.html',
                           original_file=filename,
                           error="No results returned from model")
    
def process_video(filepath, filename):
    results = model(filepath, stream=True)
    
    output_filename = 'result_' + filename
    output_filepath = os.path.join(UPLOAD_FOLDER, output_filename)
    
    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
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


@app.route('/live', methods=['GET'])
def live():
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow('Live Detection')
    cv2.namedWindow('Detected Objects')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame, show=False)
        
        if len(results) > 0:
            result = results[0]
            
            annotated_frame = result.plot()
            
            if hasattr(result, 'boxes') and len(result.obb) > 0:
                max_crops = 5  
                crop_height = 300  
                crop_width = 160   
                crops_display = np.zeros((crop_height, crop_width * max_crops, 3), dtype=np.uint8)
                
                for idx, box in enumerate(result.obb[:max_crops]):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop the object
                    cropped = frame[y1:y2, x1:x2]
                    
                    if cropped.size > 0: 
                        try:
                            # Resize crop to match the specified dimensions
                            cropped_resized = cv2.resize(cropped, (crop_width, crop_height))
                            
                            # Calculate position in crops display
                            start_x = idx * crop_width
                            crops_display[:, start_x:start_x + crop_width] = cropped_resized
                            
                            # Add text label
                            label = f"{class_name} {conf:.2f}"
                            cv2.putText(crops_display, label, 
                                      (start_x + 5, crop_height - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        except Exception as e:
                            print(f"Error processing crop {idx}: {e}")
                            continue
                
                # Show both windows separately
                cv2.imshow('Live Detection', annotated_frame)
                cv2.imshow('Detected Objects', crops_display)
            else:
                cv2.imshow('Live Detection', annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources after the loop ends
    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)