from flask import Flask, request, render_template
import os
from model.train_model import predict_image


# Kreiramo Flask aplikaciju
app = Flask(__name__)

# Podesimo folder gde će se čuvati upload-ovane slike
UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta za početnu stranicu


@app.route('/')
def home():
    return '''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Image Classifier</title>
            <link rel="stylesheet" href="/static/style.css">
        </head>
        <body>
            <div class="container">
                <h1>AI Image Classifier</h1>
                <p>Upload an image to see what the AI thinks it is!</p>
                <form method="post" action="/upload" enctype="multipart/form-data">
                    <input type="file" name="file" required>
                    <button type="submit">Upload</button>
                </form>
            </div>
        </body>
        </html>
    '''

# Ruta za upload slike i predikciju


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part in the request"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Poziv funkcije za predikciju
        predictions = predict_image(file_path)
        formatted_predictions = [(label, round(prob * 100, 2))
                                 for (_, label, prob) in predictions[0]]

        # Renderujemo HTML stranicu sa predikcijama
        return render_template(
            'result.html',
            image_url=f'/static/uploads/{file.filename}',
            predictions=formatted_predictions
        )


# Pokretanje aplikacije
if __name__ == "__main__":
    app.run(debug=True)
