import cv2
import mediapipe as mp
import numpy as np

from common import Triangle, Point

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

_PUSHUP = "push-up"
## param start
elbow_before = 180
cnt = 0
frameCnt = 0
pre_elbow = 1000
down_cnt = 0
up_cnt = 0
pre_cnt = 0
check_diff = 0
## param end


def poseDetect(pose_landmarks, image_width, image_height, video_cnt):
    """
    根据输入的身体各个部位的定位坐标，判断pose类型并计数
    """
    global up_cnt, check_diff,pre_cnt, down_cnt, pre_elbow, elbow_before, cnt, frameCnt
    threshold_t = 165  # 腰部弯曲度阈值
    threshold_h = 70  # 肩腰膝的水平高度标准偏差阈值
    threshold_wave = 20  # 运动轨迹可以类似为一个波形，这个数字为波长的阈值
    pose = "NAN"
    # 左键的三维坐标值
    LEFT_SHOULDER_x = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image_width
    LEFT_SHOULDER_y = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image_height
    LEFT_SHOULDER_z = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z * image_width
    # 腰部的三维坐标值
    LEFT_HIP_x = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image_width
    LEFT_HIP_y = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image_height
    LEFT_HIP_z = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].z * image_width
    # 膝盖的三维坐标值
    LEFT_KNEE_x = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * image_width
    LEFT_KNEE_y = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * image_height
    LEFT_KNEE_z = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].z * image_width
    # 脚踝的三维坐标值（暂时不用）
    LEFT_ANKLE_x = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].x * image_width
    LEFT_ANKLE_y = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].y * image_height
    LEFT_ANKLE_z = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE].z * image_width
    # 左肘部的三维坐标
    LEFT_ELBOW_x = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image_width
    LEFT_ELBOW_y = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image_height
    LEFT_ELBOW_z = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].z * image_width
    # 左手腕三维坐标
    LEFT_WRIST_x = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * image_width
    LEFT_WRIST_y = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * image_height
    LEFT_WRIST_z = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z * image_width
    # 肩部，腰部和膝盖三点组成的三角形，用来求出腰部的弯曲角度
    HIP_t = Triangle(Point(LEFT_SHOULDER_x, LEFT_SHOULDER_y, LEFT_SHOULDER_z),
                     Point(LEFT_HIP_x, LEFT_HIP_y, LEFT_HIP_z),
                     Point(LEFT_KNEE_x, LEFT_KNEE_y, LEFT_KNEE_z))
    # 肩部，肘部和首部三点组成的三角形，用来求出肘部的弯曲角度
    ELBOW_t = Triangle(Point(LEFT_SHOULDER_x, LEFT_SHOULDER_y, LEFT_SHOULDER_z),
                       Point(LEFT_ELBOW_x, LEFT_ELBOW_y, LEFT_ELBOW_z),
                       Point(LEFT_WRIST_x, LEFT_WRIST_y, LEFT_WRIST_z))
    # LEFT_KNEE_t = Triangle(Point(LEFT_HIP_x, LEFT_HIP_y, LEFT_HIP_z),
    #                        Point(LEFT_KNEE_x, LEFT_KNEE_y, LEFT_KNEE_z),
    #                        Point(LEFT_ANKLE_x, LEFT_ANKLE_y, LEFT_ANKLE_z))
    # 求出肩部，腰部，膝盖部三点的水平高度的标准偏差。
    # 标准偏差越大，越偏向于直立。反之越偏向于平躺
    heightArray = np.array([LEFT_SHOULDER_y, LEFT_HIP_y, LEFT_KNEE_y])
    stdInt = np.std(heightArray)
    # 腰部的弯曲角度接近于180度，并且肩部，腰部，膝盖几乎处于一个水平方向，则判定为俯卧撑姿势
    #if (HIP_t.angle_p2() > threshold_t) & (float(stdInt) < threshold_h):
    elbow = ELBOW_t.angle_p2()
    if pre_elbow != 1000:
        check_diff = max(check_diff, pre_elbow - elbow)
    pre_cnt += 1
    shoulder_p = Point(LEFT_SHOULDER_x, LEFT_SHOULDER_y, LEFT_SHOULDER_z)
    print("eb", elbow_before, "e", elbow, "pre", pre_elbow, "cnt", pre_cnt, "st", stdInt, shoulder_p.dump(), "vc", video_cnt, "df", check_diff)
    if elbow < pre_elbow:
        down_cnt += 1
    if ELBOW_t.angle_p2() > pre_elbow:
        up_cnt += 1
        if up_cnt >= 2 and down_cnt >= 2 and pre_cnt >= 3 and check_diff >= 8 and stdInt <= 100:# and elbow <= 120:
            frameCnt += 1
            print("elbow_before", elbow_before, "elbow", elbow, "frameCnt", frameCnt, "tt", HIP_t.angle_p2(), "cnt", pre_cnt)
            pre_cnt = 0
            check_diff = 0
            up_cnt = 0
        down_cnt = 0
    pre_elbow = ELBOW_t.angle_p2()
    pose = _PUSHUP
    cnt = frameCnt

    return pose, cnt


def detectFromVideo():
    # 读取视频:
    cap = cv2.VideoCapture("/Users/bytedance/Desktop/code/changqing.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 视频编解码器
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧数
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
    print(width, height)
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))  # 写入视频
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        video_cnt = 0
        while cap.isOpened():
            success, image = cap.read()
            video_cnt += 1
            if not success:
                #print("Ignoring empty camera frame.")
                break
            # height = imageOld.shape[0]
            # width = imageOld.shape[1]
            # heigtNew = round(height * 1024 / width)
            # image = cv2.resize(imageOld, (1024, heigtNew), interpolation=cv2.INTER_AREA)

            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # 把BGR图像转为RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 为了提高性能，可选择将图像标记为只读。
            image.flags.writeable = False
            # 通过姿势检测器进行检测全身坐标
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_height, image_width, channel = image.shape
            # newImg = np.zeros((image_height, image_width, channel), np.uint8)
            # 绘制面部网格
            #mp_drawing.draw_landmarks(
            #    image=image,
            #    landmark_list=results.face_landmarks,
            #    connections=mp_holistic.FACEMESH_TESSELATION,
            #    landmark_drawing_spec=None,
            #    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            # 绘制身体部位的标志点
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width
            # y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height
            if results.pose_landmarks is not None:
                # 进行俯卧撑姿势检测，以及次数统计
                pose, cnt = poseDetect(results.pose_landmarks, image_width, image_height, video_cnt)
                if pose != "NAN":
                    # 在图像中输出
                    #cv2.putText(image, pose + ":" + str(cnt), (200, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (48, 48, 255), 4)
                    #print(image_height, image_width)
                    cv2.putText(image, pose + ":" + str(cnt), (10, int(height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (48, 48, 255), 4)
            #cv2.waitKey

            out.write(image)  # 写入帧
            cv2.imshow('MyPose', image)
            #break
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cap.release()
    out.release()


if __name__ == "__main__":
    detectFromVideo()
