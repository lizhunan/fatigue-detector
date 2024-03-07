import cv2
from main import init, fatigue_detector
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ear', help="The mean aspect ratio for eyes.", default=0.2, type=float)
parser.add_argument('--mar', help="The mouth aspect ratio.", default=0.6, type=float)
parser.add_argument('--max_blinks', help="Maximum blink threshold.", default=12, type=int)
parser.add_argument('--max_yawning', help="Maximum yawning threshold.", default=6, type=int)
args = parser.parse_args()

detector, predictor = init(args.ear, args.mar, args.max_blinks, args.max_yawning)
cap = cv2.VideoCapture(0)

while(1):
    ret, origin_im = cap.read()
    cv2.imshow("capture", fatigue_detector(origin_im, detector, predictor))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 