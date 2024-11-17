import numpy as np
from keras import preprocessing
from keras import models
from keras import layers

json_file = open('model.json', 'r')
loaded_model_json =json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")
print("Model loaded from the disk")
label=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___Healthy",
       "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_",
       "Corn_(maize)___Healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
       "Grape___Esca_(Black_Measles)","Grape___Healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
       "Potato___Early_blight","Potato___Healthy","Potato___Late_blight","Tomato___Bacterial_spot",
       "Tomato___Early_blight","Tomato___Healthy","Tomato___Late_blight","Tomato___Leaf_Mold",
       "Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
       "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus"]
test_image = preprocessing.image.load_img('testing/a.blackrot.jpg',target_size=(128,128))
test_image = preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = loaded_model.predict(test_image)
print(result)
fresult =np.max(result)
label2 =label[result.argmax()]
print(label2)   