from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import pickle
import base64
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
file_path = 'feature_vectors.pkl'

with open(file_path, 'rb') as file:
    loaded_dict = pickle.load(file)

searcher = ImageSearcher(loaded_dict)

@app.route('/', methods=['GET', 'POST'])
def index():
    base64_img = ""  # Empty placeholder
    result_list = []

    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        if photo.filename != '':
            photo_data = photo.read()
            
            # Convert the uploaded image to base64 for displaying
            base64_img = base64.b64encode(photo_data).decode('utf-8')

            # Perform the search
            search_result = searcher.search(photo_data, 10)

            # Prepare the result for rendering
            for similarity, image_path in search_result:
                full_path = './preprocessed_dataset/' + image_path
                base64_image = image_path_to_base64(full_path)
                result_list.append({'image_path': base64_image, 'similarity': similarity})

    return render_template('index.html', result_list=result_list, base64_img=base64_img)

if __name__ == '__main__':
    app.run(debug=True)
