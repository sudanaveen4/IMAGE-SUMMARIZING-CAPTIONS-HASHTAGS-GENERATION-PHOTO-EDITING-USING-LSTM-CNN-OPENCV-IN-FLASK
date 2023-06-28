import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Model
import base64
import openai
from tqdm.notebook import tqdm
import cv2

app = Flask(__name__)

# Load
model_1 = VGG16()
model_1 = Model(inputs=model_1.inputs, outputs=model_1.layers[-2].output)

model = tf.keras.models.load_model('best_model.h5')

all_captions = []
with open(r'all_captions.txt', 'r') as fp:
    for line in fp:
        x = line[:-1]
        all_captions.append(x)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

max_length = max(len(caption.split()) for caption in all_captions)

max_length = max(len(caption.split()) for caption in all_captions)


def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))

    image = img_to_array(image)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    image = preprocess_input(image)
    return image


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):

    in_text = 'startseq'

    for i in range(max_length):

        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # index with high probability
        yhat = np.argmax(yhat)

        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break
    return in_text


def pre_fe(image):
    temp = model_1.predict(image, verbose=0)
    return temp

def enhance_image(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    y, u, v = cv2.split(yuv_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_y = clahe.apply(y)


    enhanced_yuv = cv2.merge((enhanced_y, u, v))


    enhanced_image = cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)


    blurred_image = cv2.GaussianBlur(enhanced_image, (0, 0), 3)
    sharpened_image = cv2.addWeighted(enhanced_image, 1.5, blurred_image, -0.5, 0)

    return sharpened_image

def perp(arg):
    match arg:
        case 0:
            return "first persion"
        case 1:
            return "thirrd person"
        case default:
            return "casual"

#mood
def mood(arg):
    match arg:
        case 0:
            return "happy"
        case 1:
            return "adventure"
        case 2:
            return "sad"
        case 3:
            return "funny"
        case 4:
            return "romantic"
        case default:
            return "trending"
def sz(arg):
    match arg:
        case 0:
            return "short"
        case default:
            return "long"

openai.api_key = "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"
def captions_gen(prmpt,prp,mod,sl):
    prompt = "generate 20 "+sz(sl)+" trending "+mood(mod)+" captions about "+prmpt+" in "+perp(prp)+"way with out hashtags"
    completion = openai.Completion.create(engine="text-davinci-003",prompt=prompt,max_tokens=1024,n=1,stop=None,temperature=0.5)
    response = completion.choices[0].text
    captionl=response.split(". ")
    caplist={}
    for i in range(1,20):
        caplist[i]=((captionl[i].split('\n')[0]))
    return caplist

def hashtags_gen(prmpt,mod):
    prompt = "generate 30 trending "+mood(mod)+" hashtags about "+prmpt+" and line by line with out numbering is required"
    hashtags = openai.Completion.create(engine="text-davinci-003",prompt=prompt,max_tokens=1024,n=1,stop=None,temperature=0.5)
    hashtags = hashtags.choices[0].text
    hashtagg=(hashtags.split("\n"))[2:]
    ras="".join(hashtagg)
    return hashtagg , ras



temp_path = 'static/temp.jpg'

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/services')
def services():
    return render_template('Services.html')

@app.route("/caption")
def caption():
    return render_template('captions.html',pred={})

@app.route("/editor")
def editor():
    return render_template('imageEnhancer.html')

@app.route("/team")
def team():
    return render_template('Team.html')

@app.route("/descriptions")
def descriptions():
    return render_template('descriptions.html')

@app.route("/contact")
def contact():
    return render_template('Contact.html')

@app.route("/upload", methods=["POST"])
def upload():


    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    image = request.files['file']
    image.save(temp_path)
    print(image)
    print('asdjflkjh')
    return render_template('captions.html', imge=temp_path, hrr=1,pred={})

@app.route("/predict", methods=["POST"])
def predict():
    


    view = request.form.get('view')

    if view == "first":
        view_inx = 0
    elif view == "second":
        view_inx = 1
    else:
        view_inx = 2

    md = request.form.get('md')

    if md == "Happy":
        md_inx = 0
    elif md == "Adventure":
        md_inx = 1
    elif md == "Sad":
        md_inx = 2
    elif md == "Funny":
        md_inx = 3
    elif md == "Romantic":
        md_inx = 4
    else:
        md_inx = 5

    size = request.form.get('size')

    if size == "short":
        size_inx = 0
    else:
        size_inx = 1

    image = preprocess_image(temp_path)

    fe = pre_fe(image)

    y_pred = predict_caption(model, fe, tokenizer, max_length)

    """os.remove(temp_path)"""

    captions=captions_gen(y_pred, view_inx, md_inx, size_inx)
    hashtags,ras=hashtags_gen(y_pred, md_inx)
    print('captions')

    return render_template('captions.html', imge=temp_path, hrr=2, pred= captions, hashs= hashtags,ras=ras)


@app.route('/enhance', methods=['POST'])
def enhance():

    uploaded_image = request.files['image']

    # Reading image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    enhanced_image = enhance_image(image)

    _, img_encoded = cv2.imencode('.png', enhanced_image)
    enhanced_image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    _, img_encoded2 = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(img_encoded2).decode('utf-8')
    
    return render_template('Enhanced.html', enhanced_image=enhanced_image_base64, up_image=image_base64, tray=1)
    

if __name__ == "__main__":
    app.run()
