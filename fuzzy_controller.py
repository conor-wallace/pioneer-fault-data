'''
input      error in orientation      Logic Rules        Linguistic          Numerical
                                   _______________       ________       _________________
                                  |               |     |        |     |                 |
----------------------->+---------| Fuzzification |---->| Engine |---->| Defuzzification |----
                        ^         |_______________|     |________|     |_________________|    |
                        |                                           ________       _______    |
                        |                                          |        |     |       |   |
                         ------------------------------------------| Sensor |<----| Plant |   |
                                                                   |________|     |_______|<---
                                                                   IMU Sensor   Pioneer Motors
'''
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
tire_pressure = ctrl.Antecedent(np.arange(0, 3, 1), 'tire_pressure')
orientation_error = ctrl.Consequent(np.arange(-34, 34, 1), 'orientation_error')

# Auto-membership function population is possible with .automf(3, 5, or 7)
tire_pressure.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
orientation_error['center'] = fuzz.trimf(orientation_error.universe, [-33, -33, 0])
orientation_error['right'] = fuzz.trimf(orientation_error.universe, [-33, 0, 33])
orientation_error['left'] = fuzz.trimf(orientation_error.universe, [0, 33, 33])

#tire_pressure['average'].view()
orientation_error.view()

rule1 = ctrl.Rule(tire_pressure['poor'], orientation_error['center'])
rule2 = ctrl.Rule(tire_pressure['average'], orientation_error['right'])
rule3 = ctrl.Rule(tire_pressure['good'], orientation_error['left'])

orientation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
orientation = ctrl.ControlSystemSimulation(orientation_ctrl)

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
orientation.input['tire_pressure'] = 1

# Crunch the numbers
orientation.compute()

print orientation.output['orientation_error']
orientation_error.view(sim=orientation)
