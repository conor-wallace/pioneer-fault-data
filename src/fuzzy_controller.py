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
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import rospy
import std_msgs
from std_msgs.msg import Float64

probability = 0.0
steering = 0.0
totalActuation = []
derivativeProb = []

# Antecedent for tire_pressure classification given from the output of the RNN
tire_pressure = ctrl.Antecedent(np.arange(0, 3, 1), 'tire_pressure')
accuracy = ctrl.Antecedent(np.arange(0, 100, 0.1), 'accuracy')
# Consequent for the expected error in orientation based off the given tire_pressure classification
orientation_error = ctrl.Consequent(np.arange(-34, 34, 1), 'orientation_error')

tire_pressure_dictionary = ['full', 'right flat', 'left flat']
accuracy_dictionary = ['low', 'medium', 'high']
orientation_dictionary = ['left', 'center', 'right']

tire_pressure.automf(names=tire_pressure_dictionary)
accuracy.automf(names=accuracy_dictionary)
orientation_error.automf(names=orientation_dictionary)

large_mf = [-33, -33, 0]
average_mf = [-33, 0, 33]
small_mf = [0, 33, 33]

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
orientation_error['right'] = fuzz.trimf(orientation_error.universe, small_mf)
orientation_error['center'] = fuzz.trimf(orientation_error.universe, average_mf)
orientation_error['left'] = fuzz.trimf(orientation_error.universe, large_mf)

# Fuzzy Rules:
# IF the tires are full THEN keep center
# IF the right tires are flat THEN keep left
# IF the left tires are flat THEN keep right
rule1 = ctrl.Rule(antecedent=(tire_pressure['left flat'] & accuracy['low']), consequent=orientation_error['center'], label='rule1')
rule2 = ctrl.Rule(antecedent=(tire_pressure['left flat'] & accuracy['medium']), consequent=orientation_error['left'], label='rule2')
rule3 = ctrl.Rule(antecedent=(tire_pressure['left flat'] & accuracy['high']), consequent=orientation_error['left'], label='rule3')
rule4 = ctrl.Rule(antecedent=(tire_pressure['right flat'] & accuracy['low']), consequent=orientation_error['center'], label='rule4')
rule5 = ctrl.Rule(antecedent=(tire_pressure['right flat'] & accuracy['medium']), consequent=orientation_error['right'], label='rule5')
rule6 = ctrl.Rule(antecedent=(tire_pressure['right flat'] & accuracy['high']), consequent=orientation_error['right'], label='rule6')
rule7 = ctrl.Rule(antecedent=(tire_pressure['full'] & accuracy['low']), consequent=orientation_error['center'], label='rule7')
rule8 = ctrl.Rule(antecedent=(tire_pressure['full'] & accuracy['medium']), consequent=orientation_error['center'], label='rule8')
rule9 = ctrl.Rule(antecedent=(tire_pressure['full'] & accuracy['high']), consequent=orientation_error['center'], label='rule9')

# Controller
orientation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
orientation = ctrl.ControlSystemSimulation(orientation_ctrl)

tire_pressure.view()
orientation_error.view()
#orientation_ctrl.view()
#accuracy.view()
plt.show()

def predCallback(data):
    global probability
    probability = data.data

def steerCallback(data):
    global steering, totalActuation
    steering = data.data
    totalActuation.append(data.data)

rospy.init_node('fuzzyController', anonymous=True)
rospy.Subscriber("/prediction", Float64, predCallback)
rospy.Subscriber("/steering", Float64, steerCallback)
compPub = rospy.Publisher("/compensation", Float64, queue_size=1, tcp_nodelay=True)
rate = rospy.Rate(5)

compMsg = std_msgs.msg.Float64()
compMsg.data = 0.0
totalProb = 0.0

while not rospy.is_shutdown():
    # Pass Antecedent input to the Controller
    # 0: center 1: right 2: left
    if(steering != -1.0):
        if(float(steering) == 0.0):
            totalProb += 1.0
            if(len(np.array(totalActuation)) > 0):
                print(len(np.array(totalActuation)))
                print(totalProb)
                instantProb = float(100)*(float(totalProb)/float(len(np.array(totalActuation)) + 3.0))
                print("Accuracy: %s%%" % instantProb)
                derivativeProb.append(instantProb)
        orientation.input['tire_pressure'] = steering
        orientation.input['accuracy'] = probability
        # Crunch the numbers
        orientation.compute()
        compensation = orientation.output['orientation_error']
        #orientation_error.view(sim=orientation)
        #plt.show()
        print("compensation in degrees: %s" % compensation)
        compMsg.data = compensation
        compPub.publish(compMsg)
        rate.sleep()
    else:
        print("finished")
        break

time = list(xrange(len(np.array(derivativeProb))))
print(time)
print(derivativeProb)
plt.plot(np.array(time), np.array(derivativeProb),'.-')
plt.title('FDS Accuracy')
plt.xlabel('timesteps')
plt.ylabel('Percent Accuracy')

plt.show()
