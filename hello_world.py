import time

import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import pickle


model= mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
q_list = []
dq_list = []
pose = None
Kp = 90
Kd = 45


with mujoco.viewer.launch_passive(model, data) as viewer:

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    viewer.sync()
    render = True
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -45
    viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])
    model.opt.gravity[2] =0


    start = time.time()

    while viewer.is_running() and time.time() - start < 7:
        step_start = time.time()


    # # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        # print(data.qpos[:7])

        #0 to 7 --> Base to Gripper (7 [Joints]+ 1 [Gripper])
        pose = data.qpos[:7].copy()
        vel = data.qvel[:7].copy()
        print(pose)

        q_list.append([pose,time.time()-start])
        dq_list.append([vel,time.time()-start])
        # print(q_list)

        '''
        PD control to the base joint to move along a sin(wt) wave where w = pi/2 and t = time.time() - start 
        '''
        data.ctrl[0] = Kp * (-np.sin(np.pi* 2*time.time() - start) - data.qpos[0]) + Kd * (0 - data.qvel[0])
        data.ctrl[1:4] = 0
        data.ctrl[5] = Kp * (np.sin(np.pi/4 * time.time() - start) - data.qpos[5]) + Kd * (0 - data.qvel[5])

        # 6 --> End-effector ; 7--> Gripper
        data.ctrl[6] = Kp * (np.sin(np.pi/2 * time.time() - start) - data.qpos[6]) + Kd * (0 - data.qvel[6])
        '''
        Keeping other joints almost no control and move as the base joint moves
        Giving data.ctrl[1:7] = 0 causes the joints to not be actuated 
        '''
        # data.ctrl[1:7] = Kp * (0 - data.qpos[1:7]) + Kd * (0-0)
        # data.ctrl[:7] = Kp * (np.sin(np.pi/2 * time.time() - start)- data.qpos[:7]) + Kd * (0-data.qvel[:7])
        # data.ctrl[1:7] = 0

        time.sleep(2e-5)

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        
        mujoco.mj_step(model, data)

with open('q_leader.pkl', 'wb') as file:

    pickle.dump(q_list, file)

with open('dq_leader.pkl', 'wb') as file:

    pickle.dump(dq_list, file)

with open('q_leader.pkl', 'rb') as file: 
    
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
        plt.plot(timestamps, joint_values_i, label=f'Joint {i}')

    # Adding labels and legend
    plt.xlabel('Timestamp')
    plt.ylabel('Joint Value')
    plt.title('Joint Values vs. Timestamp')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

