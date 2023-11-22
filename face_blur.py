import cv2 as cv
import numpy as np
import mediapipe as mp

cam_or_video_input = input("Do you want to read video directly from the camera or from a media file : (file/cam) ")

if cam_or_video_input == "file":
	print("Loading video from media file 'video_fotage.mp4'")
	capture = cv.VideoCapture("video_fotage.mp4")
else:
	print("Okay reading video directly from camera")
	capture = cv.VideoCapture(0)

x1,y1 = 0,0
x2,y2 = 0,0
drawEdges = False

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


while True:
    try:
        read, frame = capture.read()
        frame = cv.flip(frame, 1)
        
        results = faceDetection.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        
        detections = results.detections
        if detections:
            for id, faceLandmarks in enumerate(detections):
                print(id)
                print(faceLandmarks.score)
                print(faceLandmarks.location_data.relative_bounding_box)
                
                h, w, channel = frame.shape
                
                x1 = int(faceLandmarks.location_data.relative_bounding_box.xmin*w)
                y1 = int(faceLandmarks.location_data.relative_bounding_box.ymin*h)
                x2 = x1+int(faceLandmarks.location_data.relative_bounding_box.width*w)
                y2 = y1+int(faceLandmarks.location_data.relative_bounding_box.height*h)
                
                detectionConfidence = faceLandmarks.score[0]
                detectionConfidenceStr = f"{int(detectionConfidence*100)}%"
                
                spacing = 120
                cv.putText(frame, detectionConfidenceStr, (x1-spacing, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, [0,255,255], 2)
                
                #Draw thick edges
                if drawEdges:
                    lineLength = 20
                    lineColor = [0,255,0]
                    
                    cv.line(frame, (x1,y1), (x1+lineLength, y1), lineColor, 5)
                    cv.line(frame, (x1,y1), (x1, y1+lineLength), lineColor, 5)
                    
                    cv.line(frame, (x1,y2-lineLength), (x1, y2), lineColor, 5)
                    cv.line(frame, (x1,y2), (x1+lineLength, y2), lineColor, 5)
                    
                    cv.line(frame, (x2-lineLength,y1), (x2, y1), lineColor, 5)
                    cv.line(frame, (x2,y1), (x2, y1+lineLength), lineColor, 5)
                    
                    cv.line(frame, (x2-lineLength,y2), (x2, y2), lineColor, 5)
                    cv.line(frame, (x2,y2-lineLength), (x2, y2), lineColor, 5)
                
                #cv.rectangle(frame, (x1,y1), (x2,y2), (100,200,200), -1)
                
                # Apply Blur on face
                GAPX = 20
                GAPY = 20
                
                try:
                    frameROI = frame[y1-(GAPY+60):y2+GAPY,x1-(GAPX+15):x2+GAPX]
                    blur_image = cv.GaussianBlur(frameROI, (23, 23), 200)
                    frame[y1-(GAPY+60):y2+GAPY,x1-(GAPX+15):x2+GAPX] = blur_image
                except:
                    pass
            
        cv.imshow("Face Blur", frame)

        if cv.waitKey(1) & 0xFF == ord("d"):
            break
    except:
        pass

capture.release()
