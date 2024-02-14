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
pose = None


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

    while viewer.is_running() and time.time() - start < 2:
        step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

    # # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()
        # print(data.qpos[:7])
        data.ctrl[2] = np.cos(np.pi/4)  

        # print(q_list)
        time.sleep(2e-5)

        pose = data.qpos[:7].copy()
        # print(pose)

        q_list.append([pose,time.time()-start])
        # print(q_list)

with open('q.pkl', 'wb') as file:

    pickle.dump(q_list, file)

with open('q.pkl', 'rb') as file: 
    
# Call load method to deserialze 
    myvar = pickle.load(file) 

    print(myvar[0])
    # print(myvar[len(myvar)-1])
        # Extracting joint values and timestamps
    joint_values = [data[0] for data in myvar]
    timestamps = [data[1] for data in myvar]

    # Plot each joint value against its corresponding timestamp
    for i in range(len(joint_values[0])):
        joint_values_i = [joints[i] for joints in joint_values]
        plt.plot(timestamps, joint_values_i, label=f'Joint {i+1}')

    # Adding labels and legend
    plt.xlabel('Timestamp')
    plt.ylabel('Joint Value')
    plt.title('Joint Values vs. Timestamp')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

