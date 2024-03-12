import time

import mujoco
import mujoco.viewer
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math


model= mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

# target = np.array(
# [ 0.13356409 ,0.01067754, -0.01287296, -0.0694974 , -0.00348183,  0.46568362,
#  -0.06095309])

q_track = []


dq_max =[]
ddq_max = []

'''
Alignment gains
'''
k_p_follower_align = np.array([45,45,45,45,18,11,5]) # Given in the teleop_joint_pd_example_controller.h
k_d_follower_align = np.array([4.5,4.5,4.5,4.5,1.5,1.5,1]) # Given in the teleop_joint_pd_example_controller.h

'''
Max vel and Max acc for alignment
'''
dq_max_align = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1])
ddq_max_align = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
'''
Defining maximum lower and upper velocities and accelerations for each joint of the follower arm.
The velocities and accelerations vary depending on the ramp parameter for eachh joint
'''
dq_max_lower_ = np.array([0.8, 0.8, 0.8, 0.8, 2.5,2.5,2.5])
dq_max_upper_ = np.array([2,2,2,2,2.5,2.5,2.5])
ddq_max_lower_ = np.array([0.8, 0.8, 0.8, 0.8, 10.0, 10.0, 10.0]) # [rad/s^2]
ddq_max_upper_ = np.array([6.0, 6.0, 6.0, 6.0, 15.0, 20.0, 20.0]) # [rad/s^2]

'''
Drift Compensation Gains for the follower arm
'''
k_dq_= np.array([4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3])


print(dq_max_upper_[0])

'''
p and d gains for the follower arm for PD control
'''
k_p_follower = np.array([900,900,900,900,375,225,100])
k_d_follower = np.array([45,45,45,45,15,15,10])


def rampParameter(x, neg_x_asymptote , pos_x_asymptote , shift_along_x , increase_factor):

    ramp = 0.5 * (pos_x_asymptote + neg_x_asymptote -
        (pos_x_asymptote - neg_x_asymptote) * np.tanh(increase_factor * (x - shift_along_x)))
    
    return ramp

def saturateAndlimit(x_calc , x_last , x_max, dx_max , del_t):
    x_limited = []
    for i in range(7):
        del_x_max = dx_max[i] + del_t
        diff = x_calc[i] - x_last[i]
        print("Diff \t",diff)
        x_saturated = x_last[i] + max(min(diff, del_x_max), -del_x_max)
        xlimited = max(min(x_saturated,x_max[i]), -x_max[i])
        x_limited.append(xlimited)
    return x_limited


'''
q_leader file recorded from hello_world.py
'''
# with open('q_leader.pkl', 'rb') as file: 
    
#     # Call load method to deserialze 
#     myvar = pickle.load(file) 
#     # print(myvar[0])
#     joint_values = [data[0] for data in myvar]
#     timestamps = [data[1] for data in myvar]

#     q_leader_init = joint_values[0]

#     q_leader = joint_values
#     # print("Leader\t",q_leader)

'''
q_inverse_kinematics file recorded from hello_world_inverse_kinematics.py PINK
'''
with open('q_inverse_kinematics.pkl', 'rb') as file: 
    
    # Call load method to deserialze 
    myvar = pickle.load(file) 
    # print(myvar[0])
    joint_values = [data[0] for data in myvar]
    timestamps = [data[1] for data in myvar]
    # print("Length of Joint values",len(joint_values))
    q_leader_init = joint_values[0]

    q_leader = joint_values
    # print("Leader\t",q_leader)

'''
dq_leader file recorded from hello_world.py
'''
# with open('dq_leader.pkl', 'rb') as file: 
    
# # Call load method to deserialze 
#     myvar = pickle.load(file) 
#     # print(myvar[0])
#     jointVel_values = [data[0] for data in myvar]
#     # timestamps = [data[1] for data in myvar]

#     dq_leader = jointVel_values

'''
dq_inverse_kinematics file recorded from hello_world_inverse_kinematics.py PINK
'''
with open('dq_inverse_kinematics.pkl', 'rb') as file: 
    
# Call load method to deserialze 
    myvar = pickle.load(file) 
    # print(myvar[0])
    jointVel_values = [data[0] for data in myvar]
    # timestamps = [data[1] for data in myvar]

    dq_leader = jointVel_values

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

    kTolerance = 1e-2
    velocity_ramp_shift_= 0.25 # Given in the teleop_joint_pd_example_controller.h
    velocity_ramp_increase_ = 20 # Given in the teleop_joint_pd_example_controller.h

    follower_stiffness_scaling = 0.2

    dq_target_last_ = [0,0,0,0,0,0,0]
    iter = 0


    start = time.time()
    data.qpos[:7] = np.array([-2.3093, -1.5133, -2.4937, -2.7478, -2.48, 0.8521, -2.6895])

    # while viewer.is_running() and time.time() - start < 20:
    while viewer.is_running() and iter < len(joint_values):
        step_start = time.time()
        viewer.sync()

        q_follower = data.qpos[:7].copy()
        # print(q_follower)
        # break
        dq_follower = data.qvel[:7].copy()

        q_track.append([q_follower,time.time()-start])
        # print("Leader\t",q_leader_init)
        # print("Follower\t",q_follower)


        kNorm = np.abs(q_leader_init - q_follower)
        
        print("kNorm",kNorm)

        q_target_last_ = q_follower

        '''
        Determining deviation between each joint of both the arms every timestamp to track errors
        '''
        q_deviation = np.abs(q_target_last_ - q_leader[iter])
        # print(q_deviation[6])
        alignment_error_ = q_leader[iter] - q_follower
        if np.any(kNorm) > kTolerance:

            print("ROBOTS ARE NOT ALIGNED\t")
            print("GOING TO ALIGN MODE\t")

            '''
            Computing dq_unsaturated_ when NOT ALIGNED
            '''
            # target = q_leader_init
            # while np.all(kNorm) > kTolerance:
            #     error = target - q_follower

            #     print("Error\t",error)
                
            #     tau = Kp * error + Kd * (0 - dq_follower)
            #     data.ctrl[:7] = tau

            #     kNorm = np.abs(target - q_follower)
                
            #     time.sleep(2e-5)
            #     mujoco.mj_step(model, data)

            #     q_follower = data.qpos[:7].copy()

            dq_max = dq_max_align
            ddq_max = ddq_max_align
            prev_alignment_error = alignment_error_
            alignment_error_  = q_leader[iter] - q_follower
            dalignment_error = (alignment_error_ - prev_alignment_error) / (time.time() - start)

            dq_unsaturated_ = np.diag(k_p_follower_align) @ alignment_error_ + np.diag(k_d_follower_align) @ dalignment_error

        else: 
            print("ROBOTS ARE ALIGNED")

            for i in range(7):
                # print(i)
                dqmax = rampParameter(q_deviation[i], dq_max_lower_[i], dq_max_upper_[i],
                                         velocity_ramp_shift_ ,velocity_ramp_increase_)
                dq_max.append(dqmax)
                # print(dq_max)

                ddqmax = rampParameter(q_deviation[i], ddq_max_lower_[i], ddq_max_upper_[i],
                                           velocity_ramp_shift_, velocity_ramp_increase_)
                ddq_max.append(ddqmax)

            print("iter \t", iter)
            print("q_leader \t",q_leader[iter])                 
            dq_unsaturated_ = np.diag(k_dq_) @ (q_leader[iter]- q_target_last_) + dq_leader[iter]
                
            # print(np.diag(k_dq_))
            # print("q_leader - q_target",(q_leader[iter]- q_target_last_))

            # print(k_dq_.shape)
            # print(dq_unsaturated_.shape)

        print(dq_unsaturated_ ,"x_calc \t")
        print(dq_target_last_, "x_last \t")
        print(dq_max, "x_max \t")
        print(ddq_max ,"dx_max \t")

        '''
        Calculate target pose and vel for follower arm
        '''
        dq_target_ = saturateAndlimit(dq_unsaturated_, dq_target_last_, dq_max, ddq_max, time.time()-start)
        dq_target_last_ = dq_target_
        q_target_ = q_target_last_ + [dq_target*(time.time()-start) for dq_target in dq_target_]
        q_target_last_ = q_target_

        '''
        PD control for the follower arm to track the leader's motions
        '''    

        tau_follower = (follower_stiffness_scaling * np.diag(k_p_follower)) @ (q_target_ - q_follower) + (math.sqrt(follower_stiffness_scaling) * np.diag(k_d_follower)) @ (dq_target_ - dq_follower)
        
        print(tau_follower)

        data.ctrl[:7] = tau_follower

        time.sleep(2e-5)

        iter+=1
        print("iter", iter)
        # print(len(dq_max))

        if len(dq_max) == 7 and len(ddq_max) ==7:
                dq_max = []
                ddq_max = []

        mujoco.mj_step(model, data)


with open('q_follower.pkl', 'wb') as file:

    pickle.dump(q_track, file)


with open('q_follower.pkl', 'rb') as file: 
    
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


    for i in range(len(joint_values[0])):
        joint_values_i = [joints[i] for joints in joint_values]
        x = len(joint_values_i) 
        joint_values_i = np.array(joint_values_i)

        q_leader_i = [qleader[i] for qleader in q_leader ]

        q_leader_i = np.array(q_leader_i[:x])
        error = np.subtract(joint_values_i , q_leader_i)
        err_list = error.tolist()
        plt.plot(timestamps, err_list, label=f'Joint {i}')
        # plt.plot(timestamps, [joint_values_i - q_leader_i], label=f'Joint {i+1}')
    # Adding labels and legend
    plt.xlabel('Timestamp')
    plt.ylabel('Error Value')
    plt.title('Error Values vs. Timestamp')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()







        # print("Error \t" , error)
#         # error_norm = np.linalg.norm(error)
#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#         mujoco.mj_step(model, data)

#     # # Pick up changes to the physics state, apply perturbations, update options from GUI.
#         viewer.sync()
#         # print(data.qpos[:7])
#         tau = Kp * error + Kd * (0 - dq)
#         data.ctrl[:7] = tau
#         time.sleep(2e-5)
#         pose = data.qpos[:7].copy()
#         q_track.append([pose,time.time()-start])

# # print(q_track)

# with open('q_track.pkl', 'wb') as file:
#     pickle.dump(q_track, file)


# with open('q_track.pkl', 'rb') as file: 
    
# # Call load method to deserialze 
#     myvar = pickle.load(file) 

#     # print(myvar[0][0])
#     # plt.plot(myvar[])  
#     # print(myvar[len(myvar)-1])
#     # Extracting joint values and timestamps
#     joint_values = [data[0] for data in myvar]
#     timestamps = [data[1] for data in myvar]

#     # Plot each joint value against its corresponding timestamp
#     for i in range(len(joint_values[0])):
#         joint_values_i = [joints[i] for joints in joint_values]
#         plt.plot(timestamps, joint_values_i, label=f'Joint {i+1}')

#     print(joint_values[len(myvar)-1])



#     # Adding labels and legend
#     plt.xlabel('Timestamp')
#     plt.ylabel('Joint Value')
#     plt.title('Joint Values vs. Timestamp')
#     plt.legend()
#     plt.grid(True)

#     # Show the plot
#     plt.show()
