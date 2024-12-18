#!/usr/bin/env python

''' fusion.py

    Subscribe to lidar pointcloud and raw image, do transformation to match the 
    lidar points in image frame, implement camera lidar fusion based on the detect 
    target in image frame and properties of lidar pointcloud, publish clusters of 
    lidar points that associates with the detected targets.

'''

import rospy
import time
import argparse
import message_filters
import cv2
from cv_bridge import CvBridge
from cameramodels import PinholeCameraModel
from tf2_geometry_msgs import transform_to_kdl
import tf_conversions.posemath as pm
import ros_numpy
import numpy as np
from numpy.lib.function_base import median
from dnn_detect import dnn
# messages
from sensor_msgs.msg import Image,CameraInfo,PointCloud2, PointField
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs import point_cloud2
from std_msgs.msg import Header,Float32
from camera_lidar_fusion.msg import Point32Array,cluster,clusterArray

# Initialize the constants that used to do transform
lidar_to_Left_R = np.array((7.533745e-03,-9.999714e-01,-6.166020e-04,\
                            1.480249e-02,7.280733e-04,-9.998902e-01,\
                            9.998621e-01,7.523790e-03,1.480755e-02),np.float32).reshape(3,3)
lidar_to_Left_translation = np.array([-4.069766e-03,-7.631618e-02,-2.717806e-01],np.float32)


class fusion:
    def __init__(self,model):
        ''' Initialize class fusion

            Args:
                model: The path of DNN model.

            Subscribe to '/cluster_array', '/kitti/camera_color_left/image_raw'
            Publish to '/lidar_transformed', '/cluster_array'
        '''
        rospy.init_node("camera_lidar_fusion")
        self.bridge =CvBridge()
        rospy.sleep(0.5)     

        # DNN
        self.dnn_network = dnn(model)
        self.class_names = self.dnn_network.get_class_names()
        self.COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        # Camera
        self.cam_info = rospy.wait_for_message('/kitti/camera_color_left/camera_info', CameraInfo)
        self.cam_K = np.array(self.cam_info.K,dtype=np.float32).reshape(3,3)
        self.cam_D = np.array(self.cam_info.D,dtype=np.float32)
        self.cameraModel = PinholeCameraModel(self.cam_info.height,self.cam_info.width,\
                            self.cam_info.K,self.cam_info.P,self.cam_info.R,self.cam_info.D)

        # Publisher
        self.pub_lidar =  rospy.Publisher('/lidar_transformed', PointCloud2, queue_size=1)
        self.pub_Point32Array = rospy.Publisher('/cluster_array',clusterArray,queue_size=1)

        # Subscriber
        lidar_sub = message_filters.Subscriber('/kitti/velo/pointcloud', PointCloud2, queue_size=1)
        camera_sub = message_filters.Subscriber('/kitti/camera_color_left/image_raw', Image,queue_size=1)
        self.ts = message_filters.ApproximateTimeSynchronizer([lidar_sub,camera_sub],2,0.8)
        self.ts.registerCallback(self.projection)

    def projection(self,lidar_msg,cam_msg):
        ''' Callback function of the subscriber.

            Args:
                lidar_msg:  The subscribed message from the topic '/kitti/velo/pointcloud',
                            it contains all the lidar points from the rosbag.
                cam_msg: The subscribed message from the topic '/kitti/camera_color_left/image_raw',
                            it contains the raw image from the left camera from the rosbag.
        '''
        # convert camera message to readable froamt and implement DNN model
        image = self.bridge.imgmsg_to_cv2(cam_msg, desired_encoding='bgr8')
        start_drawing = time.time()
        frame,classes,scores,boxes,start,end = self.dnn_network.detect(image)
        image_width = image.shape[0]
        image_height = image.shape[1]
        copy_img = image.copy()
        
        # organize lidar pointcloud by using ros_numpy
        lidar_data = ros_numpy.numpify(lidar_msg)
        lidar_pts_xyz = np.zeros((lidar_data.shape[0],3),dtype=np.float32)
        lidar_dis = np.zeros(lidar_data.shape[0],dtype=np.float32)
        lidar_pts_xyz[:,0]=lidar_data['x']
        lidar_pts_xyz[:,1]=lidar_data['y']
        lidar_pts_xyz[:,2]=lidar_data['z']
        lidar_pts_xyz = lidar_pts_xyz[lidar_pts_xyz[:,0]>=0]   # filter out all the lidar points behind the camera
        lidar_dis = np.sqrt(np.einsum('ij,ij->i', lidar_pts_xyz[:,0:3],lidar_pts_xyz[:,0:3]))

        # lidar transformation
        lidar_pts_xyz_T = (lidar_pts_xyz.dot(lidar_to_Left_R.T))
        lidar_pts_xyz_T = lidar_pts_xyz_T - lidar_to_Left_translation
      
        # project lidar points to raw image, reshape to match the image size
        project_image =  self.cameraModel.batch_project3d_to_pixel(lidar_pts_xyz_T)
        project_image = project_image.reshape(len(project_image),2)
        project_image = np.append(project_image, lidar_dis.reshape(len(project_image),1), axis=1)

        # fliter out the lidar points outside the camera view in differnet data sets
        # keep all the points with x > 0
        lidar_dis = lidar_dis[project_image[:,0]>=0]                # depth of lidar points, same below
        lidar_pts_xyz_T = lidar_pts_xyz_T[project_image[:,0]>=0]    # transformed lidar points, same below
        lidar_pts_xyz_nonT = lidar_pts_xyz[project_image[:,0]>=0]   # original lidar points, same below
        project_image = project_image[project_image[:,0]>=0]        # lidar points in 2D space, same below

        # keep all the points with x < height of image
        lidar_dis = lidar_dis[project_image[:,0]<image_height]
        lidar_pts_xyz_T = lidar_pts_xyz_T[project_image[:,0]<image_height]
        lidar_pts_xyz_nonT = lidar_pts_xyz_nonT[project_image[:,0]<image_height]
        project_image = project_image[project_image[:,0]<image_height]
        
        # keep all the points with y < 0
        lidar_dis = lidar_dis[project_image[:,1]>=0]
        lidar_pts_xyz_T = lidar_pts_xyz_T[project_image[:,1]>=0]
        lidar_pts_xyz_nonT = lidar_pts_xyz_nonT[project_image[:,1]>=0]
        project_image = project_image[project_image[:,1]>=0]
        
        # keep all the points with y < width of image
        lidar_dis = lidar_dis[project_image[:,1]<image_width]
        lidar_pts_xyz_T = lidar_pts_xyz_T[project_image[:,1]<image_width]
        lidar_pts_xyz_nonT = lidar_pts_xyz_nonT[project_image[:,1]<image_width]
        project_image = project_image[project_image[:,1]<image_width]
        
        # cluste the lidar points target by target, and publish the clustered lidar points
        self.lidar_cluster(frame, project_image, copy_img, lidar_dis, lidar_pts_xyz_nonT, boxes, cam_msg.header)

        # reconstruct pointcloud and publish processed lidar points
        cloud_array = np.append(lidar_pts_xyz_T, lidar_dis.reshape(len(project_image),1), axis=1)
        pcheader = Header(stamp=lidar_msg.header.stamp,frame_id='camera_color_left')
        fields = [PointField('x', 0, PointField.FLOAT32,1),
                  PointField('y', 4, PointField.FLOAT32,1),
                  PointField('z', 8, PointField.FLOAT32,1),
                  PointField('intensity', 12, PointField.FLOAT32,1)]
        msg_PointCloud2 = point_cloud2.create_cloud(pcheader, fields, cloud_array)
        self.pub_lidar.publish(msg_PointCloud2)

        # Draw the boxes and scores of all the detected targets in current frame.
        for (classid, score, box) in zip(classes, scores, boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = "%s : %f" % (self.class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # put fps label, show camera lidar fusion result in one window, show minimized box in another window
        end_drawing = time.time()
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Camera Lidar Fusion", frame)
        cv2.imshow("Box Shrink",copy_img)
        cv2.waitKey(1)


    def lidar_cluster(self, frame, project_image, image, lidar_dis, lidar_pts_xyz_nonT, boxes, msg_header):
        ''' Cluste the lidar points with detected targets, remove outliers within the box by using Gaussian distribution,
            and shrink box based on the filtered lidar points.

            Args:
                frame: Current frame that captured from the video or rosbag.
                project_image: Lidar points in 2D space
                image: Raw image of camera.
                lidar_dis: Depth of lidar points
                lidar_pts_xyz_nonT: Original lidar points
                boxes: boxes: The pixel value of the detected box for all the the targets.
                msg_header: The header of the camera message.
        '''
        max_dis = np.max(lidar_dis)
        color_G = 255 / max_dis
        object_dict = {}
        lidar_xyz_cluster = {}

        # cluste lidar points by matching the box and transformed lidar points
        for index,box in enumerate(boxes):
            # top left point of box (x1,y1)
            x1 = box[0]
            y1 = box[1]

            # bottom right point if box (x2,y2)
            x2 = x1+box[2]
            y2 = y1+box[3]

            # check if the lidar points match the box of target
            lidar_pts = lidar_pts_xyz_nonT[project_image[:,0] > x1]
            pts = project_image[project_image[:,0] > x1]
            lidar_pts = lidar_pts[pts[:,0] < x2]
            pts = pts[pts[:,0] < x2]
            lidar_pts = lidar_pts[pts[:,1] > y1]
            pts = pts[pts[:,1] > y1]
            lidar_pts = lidar_pts[pts[:,1] < y2]
            pts = pts[pts[:,1] < y2]

            object_dict[index] = pts
            lidar_xyz_cluster[index] = lidar_pts

        # implement Gaussian distribution to remove outliers
        processed_dict = {}
        item_dict_dimension_pts = clusterArray(header = msg_header)
        median_dist = {}
        for key,val in object_dict.items():
            if len(val) == 0:
                continue
            
            # calculate standard deviation and mean of the depth of lidar points 
            cluster_std = np.std(val[:,2])
            cluster_mean = np.mean(val[:,2])
            
            # select data within one standard deviation around mean value
            lidar_pts = lidar_xyz_cluster[key]
            lidar_pts = lidar_pts[val[:,2]< cluster_mean + cluster_std]
            val = val[val[:,2]< cluster_mean + cluster_std]
            lidar_pts = lidar_pts[val[:,2]> cluster_mean - cluster_std]
            val = val[val[:,2]> cluster_mean - cluster_std]

            # Assign value to object cluster
            cluster_pts = cluster(header = msg_header)
            try:           
                cluster_pts.max_x = Float32(np.max(lidar_pts[:,0]))
                cluster_pts.min_x = Float32(np.min(lidar_pts[:,0]))
                cluster_pts.max_y = Float32(np.max(lidar_pts[:,1]))
                cluster_pts.min_y = Float32(np.min(lidar_pts[:,1]))
                cluster_pts.max_z = Float32(np.max(lidar_pts[:,2]))
                cluster_pts.min_z = Float32(np.min(lidar_pts[:,2]))
                cluster_pts.mean_x = Float32(np.mean(lidar_pts[:,0]))
                cluster_pts.mean_y = Float32(np.mean(lidar_pts[:,1]))
                cluster_pts.mean_z = Float32(np.mean(lidar_pts[:,2]))
            except:
                continue

            item_dict_dimension_pts.object.append(cluster_pts)
            processed_dict[key] = val

            sort_pts = val[np.lexsort((val[:,1], val[:,0]))]
            median_dist[key] = sort_pts[len(sort_pts)//2]
        
        # Draw the distance to other cars, calculated from lidar depth
        for key,val in processed_dict.items():
            color = None
            for pts in val:
                x = pts[0]
                y = pts[1]
                depth = pts[2]
                color = (0, color_G*depth, 255)
                cv2.circle(frame, (int(x), int(y)), 5, color, thickness = -1 )
            if color is not None:
                x = int(median_dist[key][0])
                y = int(median_dist[key][1])
                depth =  "{:.2f}m".format(float(median_dist[key][2]))
                cv2.putText(frame, depth, (x , y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Draw the shrunken box to raw image       
        for key,val in processed_dict.items():
            max_x = int(np.max(val[:,0]))
            min_x = int(np.min(val[:,0]))
            max_y = int(np.max(val[:,1]))
            min_y = int(np.min(val[:,1]))
            cv2.rectangle(image, (min_x,min_y),(max_x,max_y),(0, 255, 255), 2)
            x = int(median_dist[key][0])
            y = int(median_dist[key][1])
            depth =  "{:.2f}m".format(float(median_dist[key][2]))
            cv2.putText(image, depth, (x , y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
      
        # publish clustered lidar points to topic '/cluster_array'
        self.pub_Point32Array.publish(item_dict_dimension_pts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN detect')
    parser.add_argument('model',type=str,help='path to model folder')
    args, unknown = parser.parse_known_args()  # For roslaunch compatibility
    if unknown: print('Unknown args:',unknown)
    
    fusion(args.model)
    rospy.spin()


       