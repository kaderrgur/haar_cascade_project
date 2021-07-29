import cv2


vid = cv2.VideoCapture(0)

smile_cascade = cv2.CascadeClassifier('C:\\Users\\Kader\\Desktop\\PROJECT\\smile_cascade\\smile.xml')
sad_cascade = cv2.CascadeClassifier('C:\\Users\\Kader\\Desktop\\PROJECT\\sad_cascade\\sad.xml')
fear_cascade = cv2.CascadeClassifier('C:\\Users\\Kader\\Desktop\\PROJECT\\fear_cascade\\fear.xml')
face_cascade = cv2.CascadeClassifier('C:\\Users\\Kader\\Desktop\\PROJECT\\face_cascade\\frontalface.xml')


while 1:
    ret,frame = vid.read()
    frame =cv2.flip(frame,1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.5,5)
    
    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[x:x+w,y:y+h]
        roi_img = frame[x:x+w,y:y+h]

        sadness = sad_cascade.detectMultiScale(roi_gray,1.5,1)
        for (ex,ey,ew,eh) in sadness:
            cv2.rectangle(roi_img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
        smiles = smile_cascade.detectMultiScale(roi_gray,1.5,5)
        for (ex,ey,ew,eh) in smiles:
            cv2.rectangle(roi_img,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            
        scared = fear_cascade.detectMultiScale(roi_gray,1.5,9)
        for (ex,ey,ew,eh) in scared:
            cv2.rectangle(roi_img,(ex,ey),(ex+ew,ey+eh),(255,255,255),2)

    cv2.imshow('videos',frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows() 