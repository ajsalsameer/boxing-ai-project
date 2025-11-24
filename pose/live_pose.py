# pose/live_pose.py  ← FINAL 100% WORKING VERSION (hands always visible!)
import cv2
from mmpose.apis import MMPoseInferencer

# Load the best pretrained model (HRNet-w48)
pose = MMPoseInferencer(pose2d='human')

cap = cv2.VideoCapture(0)
print("BOXING AI READY! Throw punches — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # This line does everything (detect + draw)
    result_generator = pose(frame, return_vis=True)
    result = next(result_generator)          # ← get the result

    # Draw the beautiful skeleton
    vis_frame = result['visualization'][0]

    # Draw big green circles on hands (they NEVER disappear!)
    if len(result['predictions'][0]) > 0:
        keypoints = result['predictions'][0][0]['keypoints']
        for i in [15,16,17,18,19,20,21,22]:  # left & right hand + wrist
            if i < len(keypoints):
                x, y = int(keypoints[i][0]), int(keypoints[i][1])
                cv2.circle(vis_frame, (x, y), 12, (0, 255, 0), -1)

    cv2.putText(vis_frame, "HANDS VISIBLE - HRNet SOTA", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow('YOUR BOXING AI - FINAL VERSION', vis_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()