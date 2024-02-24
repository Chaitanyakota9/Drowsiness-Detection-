import cv2

# defining a video capture object
video_device = cv2.VideoCapture(1)
# sample file
face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')

while True:
    # capturing video frame by frame
    success, img = video_device.read()
    print(success)

    if success:
        face_loc = face_cascade.detectMultiScale(img, 1.1, 4)
        print(face_loc)

        for x, y, w, h in face_loc:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("My image", img)
        key = cv2.waitKey(2)
        if key == 32:
            print("Stopping")
            break

video_device.release()
cv2.destroyAllWindows()