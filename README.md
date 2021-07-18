# FaceRecognition-And-MaskDetection
 This is the 1st Module of my FYP.
 
 ## Module List
 - [1st Module](https://github.com/AbdulHadi404/FaceRecognition-And-MaskDetection)
 - [2nd Module](https://github.com/AbdulHadi404/React-Native-Attendance-App)
 - [3rd Module](https://github.com/AbdulHadi404/RFID-Attendance)
 - [Track Server](https://github.com/AbdulHadi404/track-server)

## Libraries
- cv2
- mmatplotlib
- numpy
- pandas
- pymongo
- json
- time
- datetime
- sys
- os
- glob
- tkinter

## Application Interface
<tr>
    <td align="center">
       <a href="https://www.flickr.com/photos/193485149@N02/51320450180/in/dateposted-public/" target="_blank" title="LogIn">
      <img src="https://live.staticflickr.com/65535/51320450180_5e11fe4252_z.jpg" alt="LogIn">
      </a>
    </td>
   </tr>

## Wroking
LBPH, Fisher and Eigen Faces for Face Recognition and Custom HAAR-Cascade for Mask Detection.
It also uploads data to MongoDb for Both RFID/Smart Card and Face Attendance.

### How it works
- The "Collect-Image" will take in 10 pictures of the subject.
- The "Model Train" will train those collected images to identify a subject.
- The "Recognize-Detection" opens up both the Front and Exit cameras.
- The "Save Attendance" saves both Front and Exit attendance into a .csv file.
- The "MongoSend" uploads the Face attendance records to the MongoDB.
- The "Save RFID Attendance" takes in records from an API of the phpMyAdmin database and generates a .csv file.
- The "MongoSend RFID" uploads the RFID  attendance records to the MongoDB.
