import cv2
import os
import numpy as np
def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of each person
    '''
    #print(os.listdir(root_path))
    return(os.listdir(root_path))
    

def get_class_id(root_path, train_names):
    '''
        To get a list of train images and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image in the train directories
        list
            List containing all image classes id
    '''
    faceList = []
    classList = []
    for idx, name in enumerate(train_names):
        name_path = root_path+'/'+ name
        #print(idx)
        for image in os.listdir(name_path):
            image_path = name_path+'/'+image
            #print(image_path)
            img = cv2.imread(image_path)
            faceList.append(img)
            classList.append(idx)
    return(faceList, classList)

def detect_train_faces_and_filter(image_list, image_classes_list):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered image classes id
    '''
    faceList = []
    classList = []
    for i in range(len(image_list)):
        
        imgGray = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        detected_faces = face_cascade.detectMultiScale(imgGray,scaleFactor=1.2, minNeighbors=5)
        
        if(len(detected_faces)<1):
            continue
        for face in detected_faces:
            x,y,w,h = face
            faceImg = imgGray[y:y+w, x:x+h]
            faceList.append(faceImg)
            classList.append(image_classes_list[i])
    #print(len(faceList))
    return faceList, classList

def detect_test_faces_and_filter(image_list):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
    '''
    faceList = []
    rectangle = []
    for i in range(len(image_list)):
        
        imgGray = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        detected_faces = face_cascade.detectMultiScale(imgGray,scaleFactor=1.2, minNeighbors=5)
        
        if(len(detected_faces)<1):
            continue
        for face in detected_faces:
            x,y,w,h = face
            faceImg = imgGray[y:y+w, x:x+h]
            faceList.append(faceImg)
            rectangle.append(face)
    #print(len(faceList) )
    return faceList, rectangle

def train(train_face_grays, image_classes_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_face_grays, np.array(image_classes_list))
    return recognizer

def get_test_images_data(test_root_path):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image in the test directories
    '''
    faceList = []
    for image in os.listdir(test_root_path):
        path = test_root_path + '/' + image
        img = cv2.imread(path)
        faceList.append(img)
    return faceList
def predict(recognizer, test_faces_gray):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        test_faces_gray : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    prediction = []
    for face in test_faces_gray:
        res, loss = recognizer.predict(face)
        #print(res)
        prediction.append(res)
    return prediction

def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, size):
    '''
        To draw prediction results on the given test images and resize the image

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories
        size : number
            Final size of each test image

        Returns
        -------
        list
            List containing all test images after being drawn with
            final result
    '''
    drawed_list = []
    for i in range(len(test_image_list)):
        image = test_image_list[i]

        x,y,w,h = test_faces_rects[i]

        cv2.rectangle(image,(x,y),(x+w, y+h),(255,0,0),3)
        cv2.putText(image,train_names[predict_results[i]],(x, y-10),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        image = cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
        drawed_list.append(image)
    return drawed_list

def combine_and_show_result(image_list, size):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        image_list : nparray
            Array containing image data
        size : number
            Final size of each test image
    '''
    imageV = np.hstack(image_list)
    cv2.imshow('results', imageV)
    cv2.waitKey(0)
'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''

if __name__ == "__main__":

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    train_root_path = "dataset/train"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    train_names = get_path_list(train_root_path)
    train_image_list, image_classes_list = get_class_id(train_root_path, train_names)
    train_face_grays, filtered_classes_list = detect_train_faces_and_filter(train_image_list, image_classes_list)
    recognizer = train(train_face_grays, filtered_classes_list)

    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''
    test_root_path = "dataset/test"
    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_image_list = get_test_images_data(test_root_path)
    test_faces_gray, test_faces_rects = detect_test_faces_and_filter(test_image_list)
    predict_results = predict(recognizer, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names, 200)
    
    combine_and_show_result(predicted_test_image_list, 200)