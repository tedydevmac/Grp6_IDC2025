import cv2

cam = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, image = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    cv2.imshow('Camera Feed', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
