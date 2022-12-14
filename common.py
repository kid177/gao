import numpy as np
import math
import cv2
from PIL import Image, ImageDraw, ImageFont


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def point_distance(self, p):
        #distance = math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2 + (self.z - p.z) ** 2)
        distance = math.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2)
        return distance
    def dump(self):
        return "x=%s, y=%s, z=%s" %(self.x, self.y, self.z)
        #print("x=", self.x, "y=", self.y, "z=", self.z)

class Triangle:
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.distance_p1_p2 = p1.point_distance(p2)
        self.distance_p1_p3 = p1.point_distance(p3)
        self.distance_p2_p3 = p2.point_distance(p3)

    def angle_p1(self):
        cos = (
                      self.distance_p1_p2 ** 2 + self.distance_p1_p3 ** 2 - self.distance_p2_p3 ** 2) / (
                      2 * self.distance_p1_p2 * self.distance_p1_p3)
        angle = round(np.arccos(cos) * 180 / np.pi)
        if angle >= 90:
            return 180 - angle
        else:
            return angle

    def angle_p2(self):
        cos = (
                      self.distance_p1_p2 ** 2 + self.distance_p2_p3 ** 2 - self.distance_p1_p3 ** 2) / (
                      2 * self.distance_p1_p2 * self.distance_p2_p3)
        angle = round(np.arccos(cos) * 180 / np.pi)
        return angle
        # if angle >= 90:
        #     return 180 - angle
        # else:
        #     return angle

    def angle_p3(self):
        cos = (
                      self.distance_p1_p3 ** 2 + self.distance_p2_p3 ** 2 - self.distance_p1_p2 ** 2) / (
                      2 * self.distance_p1_p3 * self.distance_p2_p3)
        angle = round(np.arccos(cos) * 180 / np.pi)
        if angle >= 90:
            return 180 - angle
        else:
            return angle


