from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import sys, numpy as np

path = Path(__file__).parent
model_file_url = 'YOUR MODEL.h5 DIRECT / RAW DOWNLOAD URL HERE!'
model_file_name = 'pretrained_model'

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = path/'models'/f'{model_file_name}.h5'
DENSENET_PATH = path/'models'/'densenet.hdf5'
IMG_FILE_SRC = '/tmp/saved_image.png'

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_model():
    # create the base pre-trained model
    base_model = DenseNet121(weights=DENSENET_PATH, include_top=False)
    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer
    predictions_layer = Dense(len(labels), activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions_layer)

    # load pretrained weights
    model.load_weights("pretrained_model.h5")

    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    with open(IMG_FILE_SRC, 'wb') as f: f.write(img_bytes)
    return model_predict(IMG_FILE_SRC, model)

def load_image(img_path, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean = np.float64(134.5683984375)
    std = np.float64(64.79917887884459)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

def model_predict(img_path, model):
    # preprocess input image
    preprocessed_input = load_image(img_path)
    # get predictions probabilites
    probs = model.predict(preprocessed_input)
    # match probs with labels and sort them
    result = sorted(zip(labels, map(float, probs[0])), key=lambda p: p[1], reverse=True)
    # print(result)
    result_html1 = path/'static'/'result1.html'
    result_html2 = path/'static'/'result2.html'
    result_html = str(result_html1.open().read() +str(result) + result_html2.open().read())
    return HTMLResponse(result_html)

@app.route("/")
def form(request):
    index_html = path/'static'/'index.html'
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
