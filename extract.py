from transform import four_point_transform
from skimage.filters import threshold_adaptive
from fuzzywuzzy import fuzz
import os
import numpy as np
from pprint import pprint
import sys
from PIL import Image
from os.path import isfile, join,isdir
import cv2
import io
import requests
import base64
from azure.storage.blob import BlockBlobService
import pyodbc
import time
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
import json
from shapely.geometry import Polygon
import re
import datetime
from fuzzysearch import find_near_matches
from fuzzywuzzy import process
from difflib import SequenceMatcher as SM
from nltk.util import ngrams
import codecs
from PIL import Image, ImageEnhance

def ImproveImage(i):
    im = Image.open("test.jpg").convert('LA')
    enhancer = ImageEnhance.Contrast(im)
    image = ImageEnhance.Sharpness(enhancer.enhance(i))
    img = image.enhance(i)
    img = img.convert("RGB")
    img.save("test.jpg")
    return None


def MatchCadena(sentence,word):
    word_length  = len(word.split())
    max_sim_val    = 0
    max_sim_string = u""

    for ngram in ngrams(sentence.split(), word_length + int(.2*word_length)):
        sentence_ngram = u" ".join(ngram)
        similarity = SM(None, sentence_ngram, word).ratio() 
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = sentence_ngram

    return max_sim_val,max_sim_string


start_time = time.time()

def IfCheckFecha(palabra):
    allm = re.findall(
        '(([0-9]|[0-3][0-9]|[0-3][0-9][0-9][0-9])[\/-]([0-9]|[0-3][0-9]|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Nomviembre|Diciembre|Jan|February|Feb|March|Mar|April|Apr|May|May|June|Jun|July|Jul|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec)[\/-]([0-9]|[0-3][0-9]|[0-3][0-9][0-9][0-9]))',
        palabra)
    if (len(allm) >= 1):
        return True
    else:
        return False



def GetDatosBasicosRegex(match):
    allm = re.findall(
        '(([0-3][0-9]|[0-3][0-9][0-9][0-9])[\/-]([0-3][0-9]|Enero|Febrero|Marzo|Abril|Mayo|Junio|Julio|Agosto|Septiembre|Nomviembre|Diciembre|Jan|February|Feb|March|Mar|April|Apr|May|May|June|Jun|July|Jul|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec)[\/-]([0-3][0-9]|[0-3][0-9][0-9][0-9]))',
        match)
    allm1 = re.findall('((S/ |S | |)+\d)', match)
    costos = [0.00]
    for x in allm1:
        costo = re.findall('\d+[\,.]+\d{1,2}', x[0])
        if (len(costo) >= 1):
        	numero = costo[0].replace(",", ".")
        	costos.append(float(numero))
    allm2 = re.findall('(\d{11})', match)
    if (len(allm2) >= 1):
        if (len(allm) >= 1):
            return allm2[0], allm[0][0], max(costos)
        else:
            return allm2[0], "", max(costos)
    else:
        if (len(allm) >= 1):
            return "", allm[0][0], max(costos)
        else:
            return "", "", max(costos)

def CorregirMonto(match):
    monto = re.findall('\d+[\:,.]+\d{1,2}', match)
    if (len(monto) >= 1):
        numero = monto[0].replace(",", ".")
        numero = monto[0].replace(":", ".")
        return  numero
    return ""


def CreatePolygon(vert,largo):
    centerX1 = 0
    centerY1 = 0
    vert = sorted(vert,key=lambda x: (x[0],x[1]))
    for i in range(0, 2):
        centerX1 += vert[i][0]
        centerY1 += vert[i][1]
    centerX1 = centerX1 / 2
    centerY1 = centerY1 / 2
    centerX = 0
    centerY = 0
    for i in range(0, 4):
        centerX += vert[i][0]
        centerY += vert[i][1]
    centerX = centerX / 4
    centerY = centerY / 4
    a = (centerY1 -centerY)/(centerX1-centerX)
    b = centerY -centerX*a
    y0 = a*largo +b
    middle = a*(largo/2) +b
    return Polygon([(0,b), (largo, y0), (0.1, b+0.1),(largo+0.1, y0+0.1)]),middle
    
def Intersectan(p1,array):
    p2 = Polygon([(array[0][0], array[0][1]), (array[1][0], array[1][1]), (array[2][0], array[2][1]),
            (array[3][0], array[3][1])])
    if (p1.intersects(p2)):
        return True
    else:
        return False
def MakeSentence(words):
    words = sorted(words,key=lambda l:l[1], reverse=False)
    sentence = ""
    for word in words:
        espacios = len(sentence)
        if(word[1]-espacios>=0):
            cant = word[1]-espacios
            cant = int(cant)
            esp = " "*(cant)
            sentence = sentence + str(esp)+word[0]
        else:
            sentence = sentence + word[0]
    return sentence.lstrip()
            


def CreateText(match_bool,vertices_palabras,descripcion,largo):
    text = []
    for i in range(0,len(vertices_palabras)):
        try:
            if(match_bool[i]== False):
                p1,centro = CreatePolygon(vertices_palabras[i],largo)
                sentence = ""
                words = []
                vertices_palabras_temp = vertices_palabras[i]
                vertices_palabras_temp = sorted(vertices_palabras_temp,key=lambda l:l[0], reverse=False)
                longitud = vertices_palabras_temp[2][0]-vertices_palabras_temp[1][0]
                espacio = longitud/float(len(descripcion[i]))
                for j in range(0,len(vertices_palabras)):
                    if(Intersectan(p1,vertices_palabras[j])):
                        match_bool[j]=True
                        vertices_palabras_temp = vertices_palabras[j]
                        vertices_palabras_temp = sorted(vertices_palabras_temp,key=lambda l:l[0], reverse=False)
                        n_espacio = round(vertices_palabras_temp[0][0]/espacio)
                        words.append([descripcion[j],n_espacio])
                sentence = MakeSentence(words)
                text.append([sentence,centro])
        except Exception as e:
            print(e)

    text = sorted(text,key=lambda l:l[1], reverse=False)
    return text
def GetStringMatch(descripcion_cortar, vertices_palabras_cortar, array):
    oracion = ""
    for i in range(0, len(descripcion_cortar)):
        p1 = Polygon([(array[0][0], array[0][1]), (array[1][0], array[1][1]), (array[2][0], array[2][1]),
                      (array[3][0], array[3][1])])
        p2 = Polygon([(vertices_palabras_cortar[i][0][0], vertices_palabras_cortar[i][0][1]),
                      (vertices_palabras_cortar[i][1][0], vertices_palabras_cortar[i][1][1]),
                      (vertices_palabras_cortar[i][2][0], vertices_palabras_cortar[i][2][1]),
                      (vertices_palabras_cortar[i][3][0], vertices_palabras_cortar[i][3][1])])
        if (p1.intersects(p2)):
            oracion = oracion + descripcion_cortar[i]
    return oracion


stemmer = LancasterStemmer()
ERROR_THRESHOLD = 0.2


# load our calculated synapse values
def Cargar_Model(synapse_file):
    with open(synapse_file) as data_file:
        synapse = json.load(data_file)
        synapse_0 = np.asarray(synapse['synapse0'])
        synapse_1 = np.asarray(synapse['synapse1'])
        words = np.asarray(synapse['words'])
        classes = np.asarray(synapse['classes'])
    return synapse, synapse_0, synapse_1, words, classes


cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=apesegprofuturo.southcentralus.cloudapp.azure.com,1433\\SQLEXPRESS;"
                      "Database=DB_OCR;"
                      "UID=Administrador;"
                      "PWD=P@ssw0rd.321;")

cursor = cnxn.cursor()


def classify(sentence, synapse, synapse_0, synapse_1, words, classes, show_details=False):
    results = think(sentence, synapse, synapse_0, synapse_1, words, classes, show_details)
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_results = [[classes[r[0]], r[1]] for r in results]
    return return_results


def CheckIfFactura(description, top, synapse, synapse_0, synapse_1, words, classes):
    for word in description:
        resultado = classify(word, synapse, synapse_0, synapse_1, words, classes)
        if (len(resultado) > 0):
            if (resultado[0][1] > top):
                return True
    return False


def GetEmpresa(description, top, synapse, synapse_0, synapse_1, words, classes):
    if(len(description)>80):
        numero = 80
    else:
        numero = len(description)
    for i in range(0,numero):
        word = description[i]
        resultado = classify(word, synapse, synapse_0, synapse_1, words, classes)
        if (len(resultado) > 0):
            if (resultado[0][1] > top):
                return resultado[0][0]
    return None


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


def think(sentence, synapse, synapse_0, synapse_1, words, classes, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def save_image(name):
    try:
        block_blob_service = BlockBlobService(account_name='gysocr',
                                              account_key='sqyfG8R0kZ22vczo7orsPeYN++9E+1PU2s+4rj47m+fTy1Y/nz+Vkr8/YWYrACo5BKy7PtnQajD1Q5MTfvUK8g==')
        block_blob_service.create_blob_from_path("images", name + ".jpg", "test.jpg")
        return ("https://gysocr.blob.core.windows.net/images/" + name + ".jpg")
    except Exception as e:
        print(e)


# file_service = FileService(account_name='myaccount', account_key='mykey')

# file_service.create_file_from_path(
#    'myshare',
#    None, # We want to create this blob in the root directory, so we specify None for the directory_name
#    'myfile',
#    'sunset.png',
#    content_settings=ContentSettings(content_type='image/png'))

import math


def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def GetPositionInvoice(vert):
    centerX = 0
    centerY = 0
    for i in range(0, 4):
        centerX += vert[i][0][0]
        centerY += vert[i][0][1]
    centerX = centerX / 4
    centerY = centerY / 4
    x0 = vert[0][0][0]
    y0 = vert[0][0][1]
    if x0 < centerX:
        if (y0 < centerY):
            return 0
        else:
            return 270
    else:
        if (y0 < centerY):
            return 90
        else:
            return 180


def convertDraw(array):
    return np.array([[array[0]], [array[3]]
                        , [array[2]], [array[1]]])


def aumentar(array, aumento):
    array = sorted(array, key=lambda x: x[1])
    array[0] = [array[0][0], array[0][1] - aumento]
    array[1] = [array[1][0], array[1][1] - aumento]
    array[2] = [array[2][0], array[2][1] + aumento]
    array[3] = [array[3][0], array[3][1] + aumento]
    array = sorted(array, key=lambda x: x[0])
    array[0] = [array[0][0] - aumento, array[0][1]]
    array[1] = [array[1][0] - aumento, array[1][1]]
    array[2] = [array[2][0] + aumento, array[2][1]]
    array[3] = [array[3][0] + aumento, array[3][1]]
    return array


def rotate(vertices_i, angle, mat):
    height, width = mat.shape[:2]
    if (angle == 270):
        return np.array(
            [[width - vertices_i[0]["y"], vertices_i[0]["x"]], [width - vertices_i[3]["y"], vertices_i[3]["x"]],
             [width - vertices_i[2]["y"], vertices_i[2]["x"]], [width - vertices_i[1]["y"], vertices_i[1]["x"]]])
    elif (angle == 180):
        return np.array(
            [[vertices_i[0]["x"], height - vertices_i[0]["y"]], [vertices_i[3]["x"], height - vertices_i[3]["y"]],
             [vertices_i[2]["x"], height - vertices_i[2]["y"]], [vertices_i[1]["x"], height - vertices_i[1]["y"]]])
    elif (angle == 90):
        return np.array(
            [[vertices_i[0]["y"], height - vertices_i[0]["x"]], [vertices_i[3]["y"], height - vertices_i[3]["x"]],
             [vertices_i[2]["y"], height - vertices_i[2]["x"]], [vertices_i[1]["y"], height - vertices_i[1]["x"]]])
    else:
        return np.array([[vertices_i[0]["x"], vertices_i[0]["y"]], [vertices_i[3]["x"], vertices_i[3]["y"]],
                         [vertices_i[2]["x"], vertices_i[2]["y"]], [vertices_i[1]["x"], vertices_i[1]["y"]]])


def ModeloCNN(file_name, model_file, label_file):
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: t})
        end = time.time()
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        labels1 = []
        results1 = []
        for i in top_k:
            labels1.append(labels[i])
            results1.append(results[i])
        return labels1, results1


def LlamadoGoogleOCR():
    url = 'https://vision.googleapis.com/v1/images:annotate?key=AIzaSyACsLE-KXNS1ZuBPvFoWiI1E_mvKCA3fd8'
    headers = {'content-type': 'application/json'}

    im = Image.open("test.jpg")
    im.save("test.jpg")

    uri = save_image("test")
    retry_flag = True
    retry_count = 0
    while retry_flag and retry_count < 5:
        try:
            r1 = requests.post(url, json={
                "requests": [{"image": {"source": {"imageUri": uri}}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]})
            retry_flag = False
        except:
            print ("Esperar 1 segundo/ Problema con GOOGLE OCR")
            retry_count = retry_count + 1
            time.sleep(1)
    if (r1.status_code == 200):
        return r1
    else:
        return None


def ReconocerImagen(name):
    imagen_1 = Image.open(name)
    imagen_1.save("test.jpg")
    r1 = LlamadoGoogleOCR()
    #lista = os.listdir("Resultados")
    original = cv2.imread("test.jpg")
    #numero = len(lista)
    #os.mkdir("Resultados/Resultado_"+str(numero))
    #cv2.imwrite("Resultados/Resultado_"+str(numero)+"/Imagen_original.jpg", original)
    if (r1 == None):
        return "No se pudo Contactar con Google"
    try:
        # Total de vertices de todas las palabras
        vertices_total = r1.json()["responses"][0]["textAnnotations"]
        descripcion_total_antes = r1.json()["responses"][0]["textAnnotations"][0]["description"]
        # Total de vertice del marco
        vertices = vertices_total[0]["boundingPoly"]["vertices"]
        # Palabras sin cortar
        descripcion_sin_cortar = []
    except Exception as e:
        return "No se encontro texto"

    #label, result = ModeloCNN("test.jpg", "tf_files/retrained_graph_factura.pb",
    #                          "tf_files/retrained_labels_factura.txt")
    # print("Resultados: " + str(label))
    # print("Puntajes: " + str(result))

    #label1, result1 = ModeloCNN("test.jpg", "tf_files/retrained_graph.pb", "tf_files/retrained_labels.txt")
    # print("Resultados: " + str(label1))
    # print("Puntajes: " + str(result1))
    ########################################
    ########################################################################
    ########################################################################
    # Validacion Orientacion
    vertices1 = vertices_total[1]["boundingPoly"]["vertices"]
    vert2 = x = np.array([[vertices1[0]["x"], vertices1[0]["y"]], [vertices1[3]["x"], vertices1[3]["y"]]
                             , [vertices1[2]["x"], vertices1[2]["y"]], [vertices1[1]["x"], vertices1[1]["y"]]])
    vert1 = convertDraw(vert2)
    angle = GetPositionInvoice(vert1)
    ################################################
    ########################################################################
    ########################################################################
    # Rotar Imagen
    image = cv2.imread("test.jpg")
    image = rotate_image(image, angle)
    #cv2.imwrite("Resultados/Resultado_"+str(numero)+"/Imagen_rotada.jpg", image)
    # Rotar marco
    ver = rotate(vertices, angle, image)
    orig = image.copy()
    orig2 = cv2.drawContours(orig, [convertDraw(ver).astype(int)], -1, (0, 255, 0), 2)
    vertices_palabras_sin_cortar = np.zeros(shape=(1, 4, 2))
    for i in range(1, len(vertices_total)):
        vertices_i = vertices_total[i]["boundingPoly"]["vertices"]
        descripcion_sin_cortar.append(vertices_total[i]["description"])
        vert_i = rotate(vertices_i, angle, image)
        orig2 = cv2.drawContours(orig2, [convertDraw(vert_i).astype(int)], -1, (0, 255, 0), 2)
        vertices_palabras_sin_cortar = np.concatenate((vertices_palabras_sin_cortar, [vert_i]))

    synapse, synapse_0, synapse_1, words, classes = Cargar_Model('models/synapses.json')
    if (CheckIfFactura(descripcion_sin_cortar, 0.85, synapse, synapse_0, synapse_1, words, classes) == False):
        return ("No es una factura")
    ver = aumentar(ver, 50)
    #cv2.imwrite("Resultados/Resultado_"+str(numero)+"/Original_letras.jpg", orig2)
    warped = four_point_transform(image, convertDraw(ver).reshape(4, 2))
    #kernel = np.ones((5, 5), np.uint8)
    #warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #warped = threshold_adaptive(warped, 251, offset=10)
    #warped = warped.astype("uint8") * 255
    cv2.imwrite("test.jpg", warped)

    ImproveImage(2)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    #lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    #l, a, b = cv2.split(lab)  # split on 3 different channels

    #l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    #lab = cv2.merge((l2,a,b))  # merge channels
    #img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    #cv2.imwrite("test.jpg",img2)
    warped = cv2.imread("test.jpg")
    warped = warped.copy()

    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    ########################################################################
    cv2.imwrite("test.jpg", warped)
    #cv2.imwrite("Resultados/Resultado_"+str(numero)+"/Comprobante_procesado.jpg", warped)

    image = warped.copy()
    image2 = image.copy()
    r1 = LlamadoGoogleOCR()

    if (r1 == None):
        return "No se pudo Contactar con Google"
    # Total de vertices de todas las palabras
    vertices_total = r1.json()["responses"][0]["textAnnotations"]
    descripcion_total = r1.json()["responses"][0]["textAnnotations"][0]["description"]
    # Total de vertice del marco
    vertices = vertices_total[0]["boundingPoly"]["vertices"]
    # Palabras sin cortar
    descripcion_cortar = []
    match_bool = []
    vertices_palabras_cortar = np.zeros(shape=(0, 4, 2))

    for i in range(1, len(vertices_total)):
        try:
            vertices_i = vertices_total[i]["boundingPoly"]["vertices"]
            descripcion_cortar.append(vertices_total[i]["description"])
            array = np.zeros(shape=(4, 2))
            array[0] = [vertices_i[0]["x"], vertices_i[0]["y"]]
            array[1] = [vertices_i[3]["x"], vertices_i[3]["y"]]
            array[2] = [vertices_i[2]["x"], vertices_i[2]["y"]]
            array[3] = [vertices_i[1]["x"], vertices_i[1]["y"]]
            image = cv2.drawContours(image, [convertDraw(array).astype(int)], -1, (0, 255, 0), 2)
            vertices_palabras_cortar = np.concatenate((vertices_palabras_cortar, [array]))
            match_bool.append(False)
        except Exception as e:
            array = np.zeros(shape=(4, 2))
            vertices_palabras_cortar = np.concatenate((vertices_palabras_cortar, [array]))
            descripcion_cortar.append("")
            match_bool.append(False)
    #cv2.imwrite("Resultados/Resultado_"+str(numero)+"/Comprobante_procesado_letras.jpg", image)

    #_, width = image2.shape[:2]
    #texto = CreateText(match_bool,vertices_palabras_cortar,descripcion_cortar,width)
 
    #with io.open("Resultados/Resultado_"+str(numero)+"/Comprobante.txt", "w", encoding="utf-8") as f:
    #    for i in range(0,len(texto)):
    #        f.write(texto[i][0]+" \r\n")
    #f.close()

    #for text in texto:
    #    MatchCadena(text[0].lower(),"raz. social")
    descripcion_cortar = [x.lower() for x in descripcion_cortar]
    extraccion = {}
    RAZON_SOCIAL = ""

    synapse, synapse_0, synapse_1, words, classes = Cargar_Model('models/synapses2.json')
    NombreEmpresa = GetEmpresa(descripcion_cortar, 0.9, synapse, synapse_0, synapse_1, words, classes)
    RUC,FECHA,MONTO=GetDatosBasicosRegex(descripcion_total)
    if (NombreEmpresa == None):        
        retry_flag = True
        retry_count = 0
        while retry_flag and retry_count < 30:
            try:
                cursor.execute("SELECT * FROM EMPRESAS WHERE RUC = '" + RUC+ "' ")
                retry_flag = False
            except:
                print ("Esperar 1 segundo/Problema con SQL")
                retry_count = retry_count + 1
                time.sleep(1)
        for row in cursor:
            NombreEmpresa = row.razon
            RAZON_SOCIAL = row.NOMBRE
        if(NombreEmpresa == None):
            return ("No pertenece a Jockey Plaza")
    else:     
        retry_flag = True
        retry_count = 0
        while retry_flag and retry_count < 30:
            try:
                cursor.execute("SELECT * FROM EMPRESAS WHERE razon = '" + NombreEmpresa+ "' ")
                retry_flag = False
            except:
                print ("Esperar 1 segundo/Problema con SQL")
                retry_count = retry_count + 1
                time.sleep(1)
        for row in cursor:
            RUC = row.RUC
            RAZON_SOCIAL = row.NOMBRE 
        #print(NombreEmpresa)
    retry_flag = True
    retry_count = 0
    while retry_flag and retry_count < 30:
        try:
            cursor.execute("SELECT * FROM IndicadoresxCampo WHERE RazonSocial = '" + NombreEmpresa + "' ")
            retry_flag = False
        except:
            print ("Esperar 1 segundo/Problema con SQL")
            retry_count = retry_count + 1
            time.sleep(1)
    extraccion["FECHA"] = FECHA
    extraccion["RAZON SOCIAL"] = RAZON_SOCIAL
    extraccion["MONEDA"] = "Nuevos Soles"
    #print("FECHA_REGEX: "+FECHA)
    for row in cursor:
        try:
            matching = []
            for s in descripcion_cortar:
                if (fuzz.ratio(row.Keyword, s) > 85):
                    matching.append(s)
            for i in range(0, len(matching)):
                position = descripcion_cortar.index(matching[i])
                vertices = vertices_palabras_cortar[position]
                image2 = cv2.drawContours(image2, [convertDraw(vertices).astype(int)], -1, (0, 255, 0), 2)
                # Obtener angulo inclinacion de palabra
                angle = math.atan2(vertices[0][1] - vertices[3][1], vertices[0][0] - vertices[3][0])
                k = math.sqrt((vertices[3][0] - vertices[2][0]) * (vertices[3][0] - vertices[2][0]) + (
                            vertices[3][1] - vertices[2][1]) * (vertices[3][1] - vertices[2][1]))
                zk = row.y2 * k
                array = np.zeros(shape=(4, 2))
                # Obtener Angulo
                angulo = row.x1
                newY0 = vertices[2][1] - math.sin(angulo) * zk
                newX0 = vertices[2][0] - math.cos(angulo) * zk
                array[0] = [newX0, newY0]
                # Obtener w3 y h3
                w3k = k * row.w3
                h3k = k * row.h3
                # Obtener Posicion de vertice 1
                newY1 = newY0 - w3k * math.sin(angle)
                newX1 = newX0 - w3k * math.cos(angle)
                array[1] = [newX1, newY1]
                # Obtener Posicion de vertice 2
                newY2 = newY1 + h3k * math.sin(angle - math.pi / 2)
                newX2 = newX1 + h3k * math.cos(angle - math.pi / 2)
                array[2] = [newX2, newY2]
                # Obtener Posicion de vertice 3
                newY3 = newY0 + h3k * math.sin(angle - math.pi / 2)
                newX3 = newX0 + h3k * math.cos(angle - math.pi / 2)
                array[3] = [newX3, newY3]
                image2 = cv2.drawContours(image2, [convertDraw(array).astype(int)], -1, (0, 255, 0), 2)
                palabra = GetStringMatch(descripcion_cortar, vertices_palabras_cortar, array)
                if( row.Campo == "MONTO"):
                    palabra = CorregirMonto(palabra)

                if( row.Campo == "FECHA"):
                    #print("FECHA_SIN_CHECK: "+palabra)
                    if(palabra == "" ):
                        extraccion[row.Campo] = FECHA
                        palabra = FECHA
                        #print(row.Campo + " " + FECHA)
                    else:
                        EsFecha = IfCheckFecha(palabra)
                        if(EsFecha == False):
                            extraccion[row.Campo] = FECHA
                            palabra = FECHA


                if palabra != "":
                    extraccion[row.Campo]=palabra
                    #print(row.Campo + " " + palabra)

        except Exception as e:
            print("Se callo aqui")
            print(e)
    if( RUC != None):
        extraccion["RUC"] = RUC
    return extraccion




# imga = cv2.drawContours(imga, [convertDraw(vert_i)], -1, (0, 255, 0), 2)


# imga = cv2.drawContours(imga, [convertDraw(vert_i)], -1, (0, 255, 0), 2)
