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

# Antecedent for tire_pressure classification given from the output of the RNN
tire_pressure = ctrl.Antecedent(np.arange(0, 3, 1), 'tire_pressure')
# Consequent for the expected error in orientation based off the given tire_pressure classification
orientation_error = ctrl.Consequent(np.arange(-34, 34, 1), 'orientation_error')

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

# Pass Antecedent input to the Controller
# 0: center 1: right 2: left
orientation.input['tire_pressure'] = 2

# Crunch the numbers
orientation.compute()

print orientation.output['orientation_error']
orientation_error.view(sim=orientation)
