'''
         input      orientation error   Logic Rules        Linguistic          Numerical                 Output
       ___________                    _______________       ________       _________________             _______
      |           |                  |               |     |        |     |                 |           |       |
------| NN Output |------->+---------| Fuzzification |---->| Engine |---->| Defuzzification |---------->| Plant |
      |___________|        ^         |_______________|     |________|     |_________________|    |      |_______|
                           |                                                                     |
                           |                                                                     |
                           -----------------------------------------------------------------------
'''
import rospy
import std_msgs
from std_msgs.msg import Float64
from fault_diagnostics.msg import Array
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Antecedent for tire_pressure classification given from the output of the RNN
tire_pressure = ctrl.Antecedent(np.arange(0, 3, 1), 'tire_pressure')
# Consequent for the expected error in orientation based off the given tire_pressure classification
orientation_error = ctrl.Consequent(np.arange(-34, 34, 1), 'orientation_error')
predictionClass = 0

# Auto-membership function
tire_pressure.automf(3)

# Membership functions to compute the compensated orientation
orientation_error['center'] = fuzz.trimf(orientation_error.universe, [-33, -33, 0])
orientation_error['left'] = fuzz.trimf(orientation_error.universe, [-33, 0, 33])
orientation_error['right'] = fuzz.trimf(orientation_error.universe, [0, 33, 33])

# Fuzzy Rules:
# IF the tires are full THEN keep center
# IF the right tires are flat THEN keep left
# IF the left tires are flat THEN keep right
rule1 = ctrl.Rule(tire_pressure['poor'], orientation_error['left'])
rule2 = ctrl.Rule(tire_pressure['average'], orientation_error['center'])
rule3 = ctrl.Rule(tire_pressure['good'], orientation_error['right'])
# note: Antecedents in skfuzzy can only have fuzzy values of 'poor', 'average', and 'good'
# poor: tires are full
# average: right tires are flat
# good: left tires are flat

# Controller
orientation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
orientation = ctrl.ControlSystemSimulation(orientation_ctrl)

def fuzzyCallback(data):
    if data.full > data.left_low and data.full > data.right_low:
        predictionClass = 0
    elif data.right_low > data.full and data.right_low > data.left_low:
        predictionClass = 1
    else:
        predictionClass = 2

rospy.init_node('fuzzyController', anonymous=True)
controlPub = rospy.Publisher('/fuzzy', Float64, queue_size=10, tcp_nodelay=True)
rospy.Subscriber("/prediction", Array, fuzzyCallback)
compensation = std_msgs.msg.Float64()
rate = rospy.Rate(10)
rate.sleep()
while not rospy.is_shutdown():
    # Pass Antecedent input to the Controller
    # 0: center 1: right 2: left
    orientation.input['tire_pressure'] = predictionClass
    # Crunch the numbers
    orientation.compute()
    compensation.data = np.float64(orientation.output['orientation_error'])
    rospy.loginfo("Compensated Degree:")
    print compensation
    compensation = 0.0
    controlPub.publish(compensation)
    rate.sleep()

'''
x_qual = np.arange(0, 11, 1)
x_serv = np.arange(0, 11, 1)
x_tip  = np.arange(0, 26, 1)

# Generate fuzzy membership functions
qual_lo = fuzz.trimf(x_qual, [0, 0, 5])
qual_md = fuzz.trimf(x_qual, [0, 5, 10])
qual_hi = fuzz.trimf(x_qual, [5, 10, 10])
serv_lo = fuzz.trimf(x_serv, [0, 0, 5])
serv_md = fuzz.trimf(x_serv, [0, 5, 10])
serv_hi = fuzz.trimf(x_serv, [5, 10, 10])
tip_lo = fuzz.trimf(x_tip, [0, 0, 13])
tip_md = fuzz.trimf(x_tip, [0, 13, 25])
tip_hi = fuzz.trimf(x_tip, [13, 25, 25])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_qual, qual_lo, 'b', linewidth=1.5, label='Bad')
ax0.plot(x_qual, qual_md, 'g', linewidth=1.5, label='Decent')
ax0.plot(x_qual, qual_hi, 'r', linewidth=1.5, label='Great')
ax0.set_title('Food quality')
ax0.legend()

ax1.plot(x_serv, serv_lo, 'b', linewidth=1.5, label='Poor')
ax1.plot(x_serv, serv_md, 'g', linewidth=1.5, label='Acceptable')
ax1.plot(x_serv, serv_hi, 'r', linewidth=1.5, label='Amazing')
ax1.set_title('Service quality')
ax1.legend()

ax2.plot(x_tip, tip_lo, 'b', linewidth=1.5, label='Low')
ax2.plot(x_tip, tip_md, 'g', linewidth=1.5, label='Medium')
ax2.plot(x_tip, tip_hi, 'r', linewidth=1.5, label='High')
ax2.set_title('Tip amount')
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
'''
