#!/usr/bin/env python

''' pub_to_3D.py

    Subscribe to clustered lidar points that associates with each target,
    process received message and project a cube for each target in 3D space.
    The association of the targets between frames was also implemented here.

'''

import cv2
import math
import numpy as np
import PyKDL
import rospy
from geometry_msgs.msg import Point, Transform, TransformStamped,Pose, Quaternion
from camera_lidar_fusion.msg import ClassID,IDArray,Point32Array,clusterArray,cluster
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_geometry_msgs import transform_to_kdl
from visualization_msgs.msg import Marker, MarkerArray
import tf_conversions.posemath as pm
from geometry_msgs.msg import Point, TransformStamped, Transform
from scipy.optimize import linear_sum_assignment
from copy import deepcopy


class cube_pub():
    def __init__(self):
        ''' Initialize class Boxes_Pub

            Subscribe to '/cluster_array'
            Publish to 'ObjectBox'
        '''
        rospy.init_node('cube_pub')
        rospy.sleep(1)
        self.prev_pts = {}
        self.max_dist =10   # Threshold for association
        self.prev_timestamp = 0
        self.counter = 0

        self.pub = rospy.Publisher('ObjectBox', MarkerArray, queue_size=1)     
        rospy.Subscriber('/cluster_array',clusterArray,self.do_association,queue_size=1)


    def do_association(self, cluster_msg):
        ''' Associate the targets between frames and pass the reuslts to the function that 
            publishs the targets in 3D space.

            Args:
                cluster_msg: The subscribed message from the topic '/cluster_array', it was
                             the clustered lidar points that associates with each target.
        '''
        # update timestamp when the rosbag looped over
        if self.prev_timestamp < cluster_msg.header.stamp.to_sec():
            self.prev_timestamp = cluster_msg.header.stamp.to_sec()
        else:
            self.prev_timestamp = 0
            self.prev_pts= {}

        # implement Hungarian algorithm
        if len(self.prev_pts) == 0:
            lst = []
            for i,object in enumerate(cluster_msg.object):
                max_x = object.max_x.data
                min_x = object.min_x.data
                max_y = object.max_y.data
                min_y = object.min_y.data
                max_z = object.max_z.data
                min_z = object.min_z.data
                mean_x = object.mean_x.data
                mean_y = object.mean_y.data
                mean_z = object.mean_z.data
                center_x = (max_x + min_x) /2
                center_y = (max_y + min_y) /2
                center_z = (max_z + min_z) /2
                self.prev_pts[i] =(center_x,center_y,center_z,True,(mean_x,mean_y,mean_z))
            
            # call this function to publish cube in type of marker
            self.cube_publish(lst)

        else:
            for k,v in self.prev_pts.items():
                self.prev_pts[k] = (v[0],v[1],v[2],False,v[4])

            dis_pair = []
            curr_pts = []

            for i,object in enumerate(cluster_msg.object):
                max_x = object.max_x.data
                min_x = object.min_x.data
                max_y = object.max_y.data
                min_y = object.min_y.data
                max_z = object.max_z.data
                min_z = object.min_z.data
                mean_x = object.mean_x.data
                mean_y = object.mean_y.data
                mean_z = object.mean_z.data
                center_x = (max_x + min_x) /2
                center_y = (max_y + min_y) /2
                center_z = (max_z + min_z) /2
                curr_pts.append((center_x,center_y,center_z,(mean_x,mean_y,mean_z)))

            for k,prev in self.prev_pts.items():
                dis = []
                for curr in curr_pts:
                    diff_dis = math.sqrt((prev[0]-curr[0])**2 + (prev[1]-curr[1])**2 + (prev[2]-curr[2])**2)
                    if diff_dis <= self.max_dist:
                        dis.append(diff_dis)
                    else:
                        dis.append(diff_dis+self.max_dist)
                convert_array = np.array(dis)
                dis_pair.append(convert_array)
            
            dis_pair = np.array(dis_pair)
            row_ind, col_ind = linear_sum_assignment(dis_pair)

            tracked_lst = []
            untracked_lst = []

            for row,col in zip(row_ind,col_ind):
                if dis_pair[row, col] >= self.max_dist:
                    untracked_lst.append((row,col))
                else:
                    tracked_lst.append((row,col))
            
            for i in tracked_lst:
                self.prev_pts[i[0]] = (curr_pts[i[1]][0],curr_pts[i[1]][1],curr_pts[i[1]][2],True,curr_pts[i[1]][3])

            untracked_lst_pts = []        
            for i in untracked_lst:
                untracked_lst_pts.append(curr_pts[i[1]])

            # call this function to publish cube in type of marker
            self.cube_publish(untracked_lst_pts)


    def cube_publish(self, untracked_lst_pts,scale_x=3,scale_y=2,scale_z=1):
        ''' Associate the targets between frames and pass the reuslts to the function that 
            publishs the targets in 3D space.

            Args:
                untracked_lst_pts: The targets that fails to associate to the targets in previous frame (new targets).
                scale_x: Length of the cude in x-axis.
                scale_y: Length of the cude in y-axis.
                scale_z: Length of the cude in z-axis.
        '''
        marks = MarkerArray()

        # create cube mark for each associated target
        for k,v in self.prev_pts.items():
            if v[3] == True:
                box = Marker()
                box.header = Header(frame_id='velo_link')
                box.type = Marker.CUBE
                box.id =  k
                box.pose.position.x = v[0]
                box.pose.position.y = v[1]
                box.pose.position.z = v[2]
                box.pose.orientation.x = 0.0
                box.pose.orientation.y = 0.0
                box.pose.orientation.z = 0.0
                box.pose.orientation.w = 1.0
                box.scale.z = scale_z
                box.scale.x = scale_x
                box.scale.y = scale_y
                box.color.a = 50.
                box.color.r = 255.  # associated cube marked as white.
                box.color.g = 255
                box.color.b = 255
                box.lifetime = rospy.Duration().from_sec(1)
                marks.markers.append(box)

         # create cube mark for each un-associated target            
        for i,p in enumerate(untracked_lst_pts):
            box = Marker()
            box.header = Header(frame_id='velo_link')
            box.type = Marker.CUBE
            box.id = i+len(self.prev_pts)
            box.pose.position.x = p[0]
            box.pose.position.y = p[1]
            box.pose.position.z = p[2]
            box.pose.orientation.x = 0.0
            box.pose.orientation.y = 0.0
            box.pose.orientation.z = 0.0
            box.pose.orientation.w = 1.0
            box.scale.z = scale_z
            box.scale.x = scale_x
            box.scale.y = scale_y
            box.color.a = 50.
            box.color.r = 255.   # un-associated cube marked as red.
            box.color.g = 0.
            box.color.b = 0.
            box.lifetime = rospy.Duration().from_sec(1)
            marks.markers.append(box)

        # Publish both associated and un-associated cube to topic '/ObjectBox'
        self.pub.publish( marks )

if __name__=='__main__':
   cube_pub()
   rospy.spin()

