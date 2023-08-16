from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import pickle
import base64
import numpy as np
import cv2
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity
from module import ImageSearcher

app = Flask(__name__)

# Set up configuration for file uploads
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

def image_path_to_base64(img_path):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Load your feature dictionary and create an ImageSearcher instance
file_path = 'mobilenet_v2.pkl'

with open(file_path, 'rb') as file:
    loaded_dict = pickle.load(file)

searcher = ImageSearcher(loaded_dict)

# Extract feature vectors and image paths from the loaded dictionary
feature_vectors = list(loaded_dict.values())
image_paths = list(loaded_dict.keys())

# Build a KDTree using the feature vectors
kdtree = cKDTree(feature_vectors)

@app.route('/', methods=['GET', 'POST'])
def index():
    base64_img = ""  # Empty placeholder
    result_list = []

    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        num_results = int(request.form['num_results'])
        if photo.filename != '':
            photo_data = photo.read()

            # Convert the uploaded image to base64 for displaying
            base64_img = base64.b64encode(photo_data).decode('utf-8')

            # Convert the photo_data to a numpy array (OpenCV image)
            nparr = np.frombuffer(photo_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Apply Haar Cascade for face detection
            faces = searcher.detect_faces(img)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                roi_color = img[y:y+h, x:x+w]
                resized = cv2.resize(roi_color, (128, 128))
                img = resized
            else:
                print("No faces detected in the image.")

            # Get the input feature vector
            input_feat = searcher.feature_extractor.extract(img)

            # Perform k-nearest neighbor search using the KDTree
            k_nearest_distances, k_nearest_indices = kdtree.query(input_feat, k=num_results)
            # Prepare the result for rendering
            for distance, idx in zip(k_nearest_distances, k_nearest_indices):
                image_path = image_paths[idx]
                similarity = 1 - distance
                full_path = './preprocessed_dataset/' + image_path
                celeb_name = (" ").join(image_path.split("_")[:-1])
                base64_image = image_path_to_base64(full_path)
                result_list.append({'image_path': base64_image, 'similarity': round(similarity, 4),
                                    'original_image_path': celeb_name})

    return render_template('index.html', result_list=result_list, base64_img=base64_img)

if __name__ == '__main__':
    app.run(debug=True)