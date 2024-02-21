import cv2
import numpy as np
import argparse
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time

from real_time_object_detection import CLASSES

def detect_objects(prototxt_path, model_path, confidence_threshold=0.2):
    # Load the model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    
    # Start the video stream
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # Start the FPS counter
    fps = FPS().start()

    # Main loop
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]

        # Preprocess the frame
        resized_image = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(resized_image, (1/127.5), (300, 300), 127.5, swapRB=True)
        net.setInput(blob)
        predictions = net.forward()

        # Loop over the predictions
        for i in np.arange(0, predictions.shape[2]):
            confidence = predictions[0, 0, i, 2]
            if confidence > confidence_threshold:
                idx = int(predictions[0, 0, i, 1])
                box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), cv2.COLOR_BAYER_BG2BGR_VNG[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv2.COLOR_BAYER_BGGR2BGR[idx], 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        fps.update()

    fps.stop()
    print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approximate FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak predictions")
    args = vars(ap.parse_args())

    detect_objects(args["prototxt"], args["model"], args["confidence"])
