import os
import tempfile
from flask import Flask, render_template, request
from utils.util import predict_image, predict_video

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence_real = None
    confidence_fake = None
    file_type = None
    error = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            file_ext = file.filename.rsplit('.', 1)[1].lower()

            try:
                # Temporary file, delete=False untuk Windows
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
                file.save(tmp.name)
                tmp.close()  # pastikan file bisa diakses oleh predict

                if file_ext in ['png', 'jpg', 'jpeg']:
                    prediction, confidence_real, confidence_fake = predict_image(tmp.name)
                    confidence_real = round(confidence_real * 100, 2)
                    confidence_fake = round(confidence_fake * 100, 2)
                    file_type = 'image'
                elif file_ext == 'mp4':
                    prediction, confidence_real, confidence_fake = predict_video(tmp.name)
                    confidence_real = round(confidence_real * 100, 2)
                    confidence_fake = round(confidence_fake * 100, 2)
                    file_type = 'video'
                else:
                    error = "Tipe file tidak dikenali."

                # Hapus file sementara setelah prediksi
                os.remove(tmp.name)
            except Exception as e:
                error = f"Terjadi kesalahan saat memproses file: {e}"
        else:
            error = "File tidak valid atau tidak didukung."

    return render_template(
        'index.html',
        prediction=prediction,
        confidence_real=confidence_real,
        confidence_fake=confidence_fake,
        file_type=file_type,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
