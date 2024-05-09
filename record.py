"""Records video with depth values to generate dataset"""

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from tf2_msgs.msg import TFMessage
import tf2_ros
import geometry_msgs.msg
import pathlib
import pyrealsense2 as rs
import torch.backends.cudnn as cudnn
from tf.transformations import quaternion_from_euler
import cv2
import numpy as np

import os
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from landmark_detection import YOLOv8_face
import time


class DataRecorder:
    def __init__(self):

        rospy.init_node("gaze_data_recorder", anonymous=True)

        # Initialize TF broadcaster
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.participant_no = None
        self.frame_no = 0

        self.depth_intrinsics = None

        yolo_model = "yolov8n-face" # yolov8n-face or yolov8-lite-s or yolov8-lite-t
        self.face_detector = YOLOv8_face(
            f"/home/kovan4/yolov8-face-landmarks-opencv-dnn/weights/{yolo_model}.onnx",
            conf_thres=0.45,
            iou_thres=0.5,
        )

        # ROS rate
        input("Press Enter to start recording")
        self.bridge = CvBridge()

        # Subscribe to the camera info topic
        self.camera_info_sub = rospy.Subscriber(
            "/camera/aligned_depth_to_color/camera_info",
            CameraInfo,
            self.intrinsics_callback,
        )

        # Subscribe to the camera topic
        self.color_sub = message_filters.Subscriber("/camera/color/image_raw", Image)

        # Subscribe to the depth topic
        self.depth_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image
        )
        ts = message_filters.TimeSynchronizer([self.color_sub, self.depth_sub], 1)
        ts.registerCallback(self.callback)

        self.save_path = "./data"
        self.image_path = self.save_path + "/images"
        self.numpy_path = self.save_path + "/numpy"

        os.makedirs(self.image_path + "/color/drawn", exist_ok=True)
        # os.makedirs(self.image_path + '/depth/', exist_ok=True)
        # os.makedirs(self.numpy_path + '/color/' , exist_ok=True)
        os.makedirs(self.numpy_path + "/depth/", exist_ok=True)
        self.content_file = open(self.save_path + "/metadata.txt", "w")
        self.content_file.write(
            "color_image_path depth_image_path camera_to_marker_vector left_eye_2d right_eye_2d midpoint_2d left_eye_3d right_eye_3d midpoint_3d\n"
        )

        # self.rate = rospy.Rate(15)
        self.rate = rospy.Rate(8)  # Set the callback rate to 10 Hz
        while not rospy.is_shutdown():
            rospy.spin()
            self.rate.sleep()

        self.content_file.close()

        self.color_sub.unregister()
        self.depth_sub.unregister()

        # create a metadata file

    def intrinsics_callback(self, data):
        print("Intrinsics callback")

        self.depth_intrinsics = rs.intrinsics()
        self.depth_intrinsics.width = data.width
        self.depth_intrinsics.height = data.height
        self.depth_intrinsics.ppx = data.K[2]
        self.depth_intrinsics.ppy = data.K[5]
        self.depth_intrinsics.fx = data.K[0]
        self.depth_intrinsics.fy = data.K[4]
        self.depth_intrinsics.model = rs.distortion.brown_conrady
        self.depth_intrinsics.coeffs = data.D

        # unsubscribe from the camera info topic
        self.camera_info_sub.unregister()

    def callback(self, color, depth):
        start_fps = time.time()
        try:
            color_image = self.bridge.imgmsg_to_cv2(color, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(
                depth, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            print(e)
        boxes, scores, classids, landmarks = self.face_detector.detect(color_image)
        if len(boxes) < 1:
            return

        dstimg = self.face_detector.draw_detections(color_image, boxes, scores, landmarks)

        for kpts in landmarks:
            # Assuming first two keypoints are for the eyes
            # Each keypoint has x, y, confidence; thus taking steps of 3
            right_eye = (int(kpts[0]), int(kpts[1]))  # First keypoint
            left_eye = (int(kpts[3]), int(kpts[4]))  # Second keypoint
            midpoint = (
                (left_eye[0] + right_eye[0]) // 2,
                (left_eye[1] + right_eye[1]) // 2,
            )
            
            print(left_eye, right_eye, midpoint)

        left_eye_depth = depth_image[left_eye[1], left_eye[0]]
        right_eye_depth = depth_image[right_eye[1], right_eye[0]]
        midpoint_depth = depth_image[midpoint[1], midpoint[0]]
        
        # Convert pixel coordinates to 3D coordinates
        left_eye3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [left_eye[0], left_eye[1]], left_eye_depth)
        right_eye3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [right_eye[0], right_eye[1]], right_eye_depth)
        midpoint_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [midpoint[0], midpoint[1]], midpoint_depth)

        # Record every frame as frame_no_depth.png and frame_no_color.png
        # and record the color and depth values as frame_no_color.npy and frame_no_depth.npy

        # lookup for the "marker" frame but do not wait
        try:
            camera_to_marker = self.tf_buffer.lookup_transform('camera_aligned_depth_to_color_frame', "burayi_degis", rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print("Failed to lookup transform: ", e)
            return

        depth_np = np.asanyarray(depth_image)
        depth_path = self.numpy_path + "/depth/" + f"{self.frame_no}_depth.npy"
        color_im_path = self.image_path + "/color/" + f"{self.frame_no}_color.png"
        color_im2_path = self.image_path + "/color/drawn/" + f"{self.frame_no}_color.png"
        cv2.imwrite(color_im_path, color_image)
        cv2.imwrite(color_im2_path, dstimg)
        np.save(depth_path, depth_np)


        # image path, depth path, camera to marker vector separated by commas
        content = (
            f"{color_im_path} {depth_path} "
            f"{camera_to_marker.transform.translation.x},"
            f"{camera_to_marker.transform.translation.y},"
            f"{camera_to_marker.transform.translation.z} "
            f"{left_eye[0]},{left_eye[1]} "
            f"{right_eye[0]},{right_eye[1]} "
            f"{midpoint[0]},{midpoint[1]} "
            f"{left_eye3d[0]/1000},{left_eye3d[1]/1000},{left_eye3d[2]/1000} "
            f"{right_eye3d[0]/1000},{right_eye3d[1]/1000},{right_eye3d[2]/1000} "
            f"{midpoint_3d[0]/1000},{midpoint_3d[1]/1000},{midpoint_3d[2]/1000}\n"
        )
        
        # write to metadata file
        self.content_file.write(content)

        self.frame_no += 1
        myFPS = 1.0 / (time.time() - start_fps)
        print("FPS: {:.1f}".format(myFPS))


if __name__ == "__main__":
    DataRecorder()
