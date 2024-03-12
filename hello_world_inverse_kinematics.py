import time
import pickle
import matplotlib.pyplot as plt

import math
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

import numpy as np

import qpsolvers
from loop_rate_limiters import RateLimiter

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
# from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer


import os


q_path = []
dq_path = []

def getDataPath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir

package_directory = getDataPath() #/home/pranay/franka_mujoco

robot_URDF = package_directory + "/urdf/{}.urdf".format("fr3")
robot = RobotWrapper.BuildFromURDF(robot_URDF, package_directory)

viz = start_meshcat_visualizer(robot)

#End-effector = Joint 6 = -4 
joint_name = robot.model.names[-4]

# print("Upper Joint Limits",robot.model.upperPositionLimit)
# print("Lower Joint Limits",robot.model.lowerPositionLimit)

# robot.q0 = pin.neutral(robot.model)
# robot.q0 = [-2.3093 ,-1.5133,-2.4937 ,-2.7478 ,-2.4800, 0.8521, -2.6895]
robot.q0 = robot.model.lowerPositionLimit

print(robot.q0)
print(joint_name)

end_effector_task = FrameTask(
        str(joint_name),
        position_cost=1.0,  # [cost] / [m]
        orientation_cost=1.0,  # [cost] / [rad]
        lm_damping=1,  # tuned for this setup
    )

posture_task = PostureTask(
        cost=1e-3,  # [cost] / [rad]
    )

tasks = [end_effector_task, posture_task]


configuration = pink.Configuration(robot.model,robot.data , robot.q0)
# configuration.check_limits(1)

for task in tasks:
    task.set_target_from_configuration(configuration)

# print(configuration.q)

viewer = viz.viewer

solver = qpsolvers.available_solvers[0]
if "quadprog" in qpsolvers.available_solvers:
    solver = "quadprog"

rate = RateLimiter(frequency=200.0)
dt = rate.period
# t = 0.0  # [s]
i = 0
start = time.time()

while time.time() - start < 20:
    end_effector_target = end_effector_task.transform_target_to_world
    # end_effector_target.translation[1] = np.sin(time.time()- start)

    c = np.cos(time.time()-start)
    s = np.sin(time.time()-start)

    end_effector_target.translation[0] = 0
    end_effector_target.translation[1] =  0
    end_effector_target.translation[2] = 1

    # Rotation along z matrix
    end_effector_target.rotation = np.array([[c , -s , 0],[s , c , 0],[0,0,1]])
    
    # Update visualization frames
    viewer["end_effector_target"].set_transform(end_effector_target.np)
    viewer["end_effector"].set_transform(
        configuration.get_transform_frame_to_world(
            end_effector_task.frame
        ).np
    )


    velocity = solve_ik(configuration, tasks, dt, solver=solver)
    # print(velocity)
    configuration.integrate_inplace(velocity, dt)

    pose = configuration.q[:7]
    dq = velocity[:7]
    q_path.append([pose,time.time()-start])
    dq_path.append([dq,time.time()-start])

    print(pose)
    # print(dq)

    viz.display(configuration.q)
    print(i)
    i+=1

    rate.sleep()

    # t += dt

with open('q_inverse_kinematics.pkl', 'wb') as file:

    pickle.dump(q_path, file)

with open('dq_inverse_kinematics.pkl', 'wb') as file:

    pickle.dump(dq_path, file)    

with open('q_inverse_kinematics.pkl', 'rb') as file: 
    
# Call load method to deserialze 
    myvar = pickle.load(file) 

    # print(myvar[0])
    # print(myvar[len(myvar)-1])
        # Extracting joint values and timestamps
    joint_values = [data[0] for data in myvar]
    timestamps = [data[1] for data in myvar]
    print(joint_values[0], timestamps[0])
    # Plot each joint value against its corresponding timestamp
    for i in range(len(joint_values[0])):
        joint_values_i = [joints[i] for joints in joint_values]
        # q_leader_i = [qleader[i] for qleader in q_leader ]
        plt.plot(timestamps, joint_values_i, label=f'Joint {i}')
        # plt.plot(timestamps, [joint_values_i - q_leader_i], label=f'Joint {i+1}')
    # Adding labels and legend
    plt.xlabel('Timestamp')
    plt.ylabel('Joint Value')
    plt.title('Joint Values vs. Timestamp')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()    