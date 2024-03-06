import time


import math
import pinocchio as pin
import numpy as np

import pink
from pinocchio.robot_wrapper import RobotWrapper

import os


ee_path = []



def getDataPath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir

start = time.time()
i = 0

# while time.time() - start < 5:

#     traj = np.sin(time.time()-start)
#     ee_path.append(traj)
#     i+=1

package_directory = getDataPath() #/home/pranay/franka_mujoco

robot_URDF = package_directory + "/urdf/{}.urdf".format("fr3")
robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

#End-effector = Joint 6 = -4 
joint_name = robot.model.names[-4]

print(joint_name)
