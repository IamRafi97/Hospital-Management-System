import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
from csv import DictWriter
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import csv 

window = tk.Tk()

window.title("Face Recogniser By ID")



window.configure(background='pink')



tk.Label(relief=tk.GROOVE, fg="red", bg="white", text="Patient Information System by Id", font=("times new roman",20,"bold"), width=40).grid(pady=40, column=1, row=1)
tk.Label(relief=tk.GROOVE, fg="red", bg="white", text="Enter Patient ID", font=("times new roman",20,"bold"), width=30).grid(pady=20, column=1, row=2) 


txt=tk.Entry(window,width=20 ,bg="red" ,fg="white",font=('times', 15, ' bold '))
txt.grid(pady=20, column=2, row=2)

tk.Label(relief=tk.GROOVE, fg="red", bg="white", text="Notification", font=("times new roman",20,"bold"), width=30).grid(pady=20, column=1, row=3) 

message = tk.Label(window, text="" ,bg="white"  ,fg="red"  ,width=30  ,height=2, activebackground = "brown" ,font=('times', 15, ' bold ')) 
message.grid(pady=20, column=2, row=3)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
  
    if(is_number(Id)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret,img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('patientimage',img)
            #wait for 100 miliseconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id 
        
        with open('PatientDetails\PatientDetails.csv','a+') as f:
            dict_writer = DictWriter(f,fieldnames=['Id'] )
            if os.stat('PatientDetails\PatientDetails.csv').st_size==0:
                                     dict_writer.writeheader()

            dict_writer.writerow({
                'Id' : Id,
                })
        f.close()
        message.configure(text= res)
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[0])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv('PatientDetails\PatientDetails.csv')
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
  

    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.3,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                
                
                tt=str(Id)
               
               
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
         
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    
    
    
    cam.release()
    cv2.destroyAllWindows()
   
  
tk.Button(width=40, relief=tk.GROOVE, fg="red", bg="white", text="Take Image", font=("times new roman",15,"bold"), command= TakeImages).grid(pady=15, column=1, row=4)
tk.Button(width=40, relief=tk.GROOVE, fg="red", bg="white", text="Train Image", font=("times new roman",15,"bold"), command=TrainImages).grid(pady=15, column=1, row=5)
tk.Button(width=40, relief=tk.GROOVE, fg="red", bg="white", text="Track Image", font=("times new roman",15,"bold"), command=TrackImages).grid(pady=15, column=1, row=6)
tk.Button(width=40, relief=tk.GROOVE, fg="red", bg="white", text="Quit", font=("times new roman",15,"bold"), command=window.destroy).grid(pady=15, column=1, row=7)
tk.Label(relief=tk.GROOVE, fg="red", bg="white", text="Developed by Rafi and Mrignako", font=("times new roman",20,"bold"), width=30).grid(pady=20, column=1, row=9) 

window.mainloop()
