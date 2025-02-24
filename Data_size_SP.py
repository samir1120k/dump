import pandas as pd
import numpy as np
import random
import math
from numpy import random
import cmath
import matplotlib.pyplot as plt
import multiprocessing

# Load datasets related to Base Stations, UAVs, and Clients
base = pd.read_csv(r'BS_data.csv')
uav = pd.read_csv(r'UAV_data.csv')
people = pd.read_csv(r'people_data.csv')
IRS=pd.read_csv(r'IRS_data.csv')
p_km_UP=pd.read_csv(r'P_km_up.csv')

Angle_df=pd.read_csv(r'Angle.csv') # number of IRS is 500 store in each column
h_l_km_df=pd.read_csv(r'h_l_km.csv') # number of IRS is 500 store in each column
h_l_m_df=pd.read_csv(r'h_l_m.csv') # number of IRS is 500 store in each column

Angle_UP_df=pd.read_csv(r'Angle1.csv') # number of IRS is 500 store in each column
g_l_km_df=pd.read_csv(r'h_l_km1.csv') # number of IRS is 500 store in each column
g_l_m_df=pd.read_csv(r'h_l_m1.csv') # number of IRS is 500 store in each column # corrected filename

Angle_har_df=pd.read_csv(r'Angle2.csv') # number of IRS is 500 store in each column
f_l_km_df=pd.read_csv(r'h_l_km2.csv') # number of IRS is 500 store in each column
f_l_m_df=pd.read_csv(r'h_l_m2.csv') # number of IRS is 500 store in each column # corrected filename
f_km1=pd.read_csv(r'f_km.csv')

# Constants
Wl_value = 35.28
H_value= 20
P_m_har = base['P_m_har']
T_m_har = base['T_m_har']
P_m_down = base['P_m_down']
f_km=f_km1['0']
V_lm_vfly = uav['V_lm_vfly']
V_lm_hfly = uav['V_lm_hfly']
D_l_hfly_value = 100

P_km_up=p_km_UP['0']
p_max=10 # moved inside loop
p_km_max=10
T_m=10
# D_m_current=0.49


# Additional constants for calculations
delta = 0.012
Ar = 0.1256
s = 0.05
Nr = 4
V_tip = 102
Cd = 0.022
Af = 0.2113
D_km = 0.5
# Dm=0.49
B=10 #MHz
sigma_km=10**(-13)
eta=10
kappa=0.5
num_population=50
Bh = (1 - 2.2558 * pow(10, -5) *H_value)**4.2577
# Bh = max(1, Bh)
p_l_b = (delta / 8) * Bh * Ar * s * pow(V_tip, 3)

# Determine the maximum possible rows based on the smallest dataframe size
min_rows = min(len(Angle_df), len(h_l_km_df), len(h_l_m_df), len(Angle_UP_df), len(g_l_km_df), len(g_l_m_df), len(Angle_har_df), len(f_l_km_df), len(f_l_m_df), len(f_km1))
num_rows_data_files = min_rows # Dynamically set num_rows_data_files based on the smallest dataframe length
population_size = min(50, num_rows_data_files) # Ensure population_size is not larger than available data

# Fitness function to calculate total energy consumption
def Fitness(E_ml_har, E_ml_down, E_ml_UAV):
    return E_ml_har + E_ml_down + E_ml_UAV

# Energy consumption of the UAV-IRS
def E_ml_UAV(P_l_vfly, T_l_vfly, P_lm_hfly, T_l_hfly, P_l_hov, T_lm_hov):
    return P_l_vfly * T_l_vfly + P_lm_hfly * T_l_hfly + P_l_hov * T_lm_hov

# Power calculations for different flight modes
def P_l_vfly(Wl, V_l_vfly, P_l_b, Nr, Ar, Bh):
    temp2 = Nr * Bh * Ar
    temp3 = np.sqrt(V_l_vfly**2 + (2 * Wl) / temp2)
    return ((Wl / 2) * (V_l_vfly + temp3)) + Nr * P_l_b

def P_lm_hfly(P_lm_blade, P_lm_fuselage, P_lm_induced):
    return P_lm_blade + P_lm_fuselage + P_lm_induced

def P_lm_blade(Nr, P_l_b, V_tip, V_lm_hfly):
    return Nr * P_l_b * (1 + ((3 * (V_lm_hfly**2)) / pow(V_tip, 2)))

def P_lm_fuselage(Cd, Af, Bh, V_lm_hfly):
    return (1 / 2) * Cd * Af * Bh * (V_lm_hfly**3)

def P_lm_induced(Nr, Bh, Ar, Wl, V_lm_hfly):
    return Wl * ((np.sqrt((Wl**2) / (4 * (Nr**2) * (Bh**2) * (Ar**2)) + ((V_lm_hfly**4) / 4)) - ((V_lm_hfly**2) / 2))**(1 / 2))

def P_l_hov(Wl, P_l_b, Nr, Ar, Bh):
    temp1 = Nr * P_l_b
    temp3 = np.sqrt(2 * (Nr * Bh * Ar))
    temp4 = ((Wl)**3 / 2) / temp3
    return temp1 + temp4

def T_lm_hov(T_km_com, T_kml_up, T_ml_down):
    return T_km_com + T_kml_up + T_ml_down

def R_ml_down(B,P_m_down,h_ml_worst): #eqation number 7
    temp1=(h_ml_worst*P_m_down) # Consider if min is the correct aggregation. It should be multiplication
    if (1+temp1) <= 0:
        return 0  # Return 0 if log argument is non-positive to avoid error
    return B*math.log2(1+temp1)

def h_ml_worst(h_kml_down,sigma_km): #eqation number 8
    return h_kml_down/(sigma_km) # it will return the sigal value which is minimum of all
        # the value for each itaration

def calculate_exp_i_theta(theta): # part of equation 8
  return cmath.exp(1j * theta)
 # 1j represents the imaginary unit in Python

def h_kml_down(Angle,h_l_m,h_l_km): # part of equation 8
    result=[]
    if isinstance(Angle, float): # Check if Angle is float, if so, return 0 or handle appropriately
        return 0 # Or raise an exception or return a default value as needed

    if not isinstance(Angle, pd.Series): # added check to handle non-series input
        raise TypeError(f"Expected Angle to be pd.Series, got {type(Angle)}")

    for i in range(len(Angle)):
        theta_radians = math.radians(Angle.iloc[i]) # Use iloc for position-based indexing
        results= calculate_exp_i_theta(theta_radians)
        result.append(results)

    diagonal=np.diag(result)
    # Ensure h_l_m and h_l_km are correctly formatted as numpy arrays
    h_l_m_np = h_l_m.to_numpy() # Convert Series to numpy array
    h_l_km_np = h_l_km.to_numpy() # Convert Series to numpy array
    if h_l_m_np.ndim == 1:
        h_l_m_np = h_l_m_np.reshape(-1, 1) # Reshape to 2D if necessary
    if h_l_km_np.ndim == 1:
        h_l_km_np = h_l_km_np.reshape(1, -1) # Reshape to 2D if necessary


    a=np.dot(h_l_km_np,diagonal) # Use numpy arrays for dot product
    b=np.dot(a,h_l_m_np)      # Use numpy arrays for dot product
    final=abs(b[0][0]) # Take absolute value and ensure it's a scalar
    return (final**2)

def R_kml_up(B,P_km_up,h_kml_up,Sub,sigma_m): #eqation number 4
    temp1=(P_km_up*h_kml_up)/ (Sub+(sigma_m))
    return B*math.log2(1+temp1)
#this is inside the equation 4 have to take summation of h_i_up and P_i_up
def sub(P_i_up,h_il_up):
    return P_i_up*h_il_up

def E_km_com(f_km,T_km_com):
    return eta*(10**(-28))*(f_km**3)*T_km_com

def E_kml_up(P_km_up,T_km_up):
    return P_km_up*T_km_up

def E_kml_har(P_m_har,T_m_har,h_km_har):
    return kappa*P_m_har*T_m_har*h_km_har

num_bs = 5
num_irs_ele=50
num_generation = 30 # Number of generations, increased for GA to evolve
num_uav_irs = 8
population_size = 50 # Population size for GA

# Define keys that should be subjected to crossover and mutation (numerical parameters)
numerical_keys_for_crossover = [
    'P_m_down_value', 'P_m_har_value', 'T_m_har_value',
    'f_km_value', 'V_lm_vfly_value', 'V_lm_hfly_value',
    'P_km_up_value','Angle1_row','Angle_row',
    'Angle2_row',
]

fitness_sums_GA_IRS= [] # Store sum of fitness values for each p_max


def GA_IRS(D_m_current): # Define function to process each p_max value
    print(f"calculaiton for D_m_current for GA",D_m_current)
    all_best_combinations = []
    all_best_individuals = []
    sum_fitness_current_p_max = 0 # Initialize sum of fitness for current p_max

    # Main Genetic Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        # Select unique row indices for the current BS

        index_list = list(range(num_rows_data_files)) # Create a list of all indices
        random.shuffle(index_list)
        unique_row_indices = index_list[:num_irs_ele]
        # Create dataframes with uniquely selected rows for the current BS
        # unique_row_indices = range(0,50)
        h_l_km_df_bs = h_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        # Corrected line: Ensure indices are within bounds of P_km_up
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)


        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}
            population = []
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value=0
            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)

            # Initialize population
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            T_l_vfly_value = H_value / V_lm_vfly_value
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)

            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)): # Using length of valid_indices
                f_km_value = f_km_bs[random.randint(0,population_size)] # Use BS-specific f_km
                P_km_up_value = P_km_up_bs[random.randint(0,population_size)] # Use BS-specific P_km_up

                Angle_row = Angle_df.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific Angle_df
                h_l_m_row = h_l_m_df.iloc[k, :] # Use BS-specific h_l_m_df
                h_l_km_row = h_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific h_l_km_df
                Angle1_row = Angle_UP_df.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific Angle_UP_df
                g_l_m_row = g_l_m_df.iloc[k, :] # Use BS-specific g_l_m_df
                g_l_km_row = g_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific g_l_km_df
                Angle2_row = Angle_har_df.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific Angle_har_df
                f_l_m_row = f_l_m_df.iloc[k, :] # Use BS-specific f_l_m_df
                f_l_km_row = f_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific f_l_km_df

                # Calculate power values


                # Calculate time and energy values
                # Corrected: D_l_hfly / V_lm_hfly
                E_ml_har_value = P_m_har_value * T_m_har_value
                h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Pass Series
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=D_m_current/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=D_m_current/R_kml_up_value # equation number 5
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

                # Calculate fitness
                result_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                # Store initial population data
                population.append({
                    'fitness': result_fitness,
                    'data':  {
                        'P_m_down_value': P_m_down_value,
                        'P_m_har_value': P_m_har_value,
                        'T_m_har_value': T_m_har_value,
                        'f_km_value': f_km_value,
                        'V_lm_vfly_value': V_lm_vfly_value,
                        'V_lm_hfly_value': V_lm_hfly_value,
                        'P_km_up_value':P_km_up_value,
                        'Angle1_row':Angle1_row,
                        'Angle_row':Angle_row,
                        'Angle2_row': Angle2_row,
                        }
                    })

            generations_data = []
            for j in range(num_generation):
                child_population = []
                # Corrected loop range to use valid_indices length
                for x in range(0, len(valid_indices), 2): # Loop through population with step of 2
                    if x + 1 >= len(valid_indices): # Check if i+1 is within bounds, if not break to avoid error in accessing population[i+1]
                        break
                    # Crossover
                    ranodmpopulation=[]
                    for i in range(10):
                        ranodmpopulation.append(random.choice(population))
                    ranodmpopulation = sorted(ranodmpopulation, key=lambda x: x['fitness'])
                    parent1 = ranodmpopulation[0]
                    parent2 = ranodmpopulation[1]
                    child_data = {}
                    for key in parent1['data']:
                        if key in numerical_keys_for_crossover:
                            if key in ['Angle1_row','Angle_row','Angle2_row']: # Handle Angle Series
                                child_data[key] = pd.Series(index=Angle_df.columns, dtype='float64') # Initialize empty Series for child
                                for col in Angle_df.columns: # Iterate through each column (angle direction)
                                    child_data[key][col] = float(parent1['data'][key][col]) * 0.6 + float(parent2['data'][key][col]) * (1 - 0.6)
                            else: # Handle other numerical values as before
                                child_data[key] = float(parent1['data'][key]) * 0.6 + float(parent2['data'][key]) * (1 - 0.6)
                        else:
                            child_data[key] = float(parent1['data'][key]) * 0.6 + float(parent2['data'][key]) * (1 - 0.6)

                    u = np.random.uniform(0, 1, 1)[0]
                    P_mutation = 0.5
                    if u < P_mutation:
                        for key in numerical_keys_for_crossover: # Apply mutation only to numerical keys
                            if key in ['Angle1_row','Angle_row','Angle2_row']: # Handle Angle Series
                                # child_data[key] = pd.Series(index=Angle_df.columns, dtype='float64') # Initialize empty Series for child
                                for col in Angle_df.columns: # Iterate through each column (angle direction)
                                    child_data[key][col] += random.normal(loc=0, scale=1, size=(1))[0]
                            else:
                                child_data[key] += random.normal(loc=0, scale=1, size=(1))[0] # Reduced scale for smaller perturbations in HC

                    # Compute child fitness
                    def compute_fitness(data):
                        P_m_down_value = data['P_m_down_value']
                        P_m_har_value = data['P_m_har_value']
                        T_m_har_value = data['T_m_har_value']
                        f_km_value = data['f_km_value']
                        V_lm_vfly_value = data['V_lm_vfly_value']
                        V_lm_hfly_value = data['V_lm_hfly_value']
                        P_km_up_value=data['P_km_up_value']
                        Angle_row = data['Angle_row'] # Retrieve angle row from child data
                        Angle1_row = data['Angle1_row'] # Retrieve angle row from child data
                        Angle2_row = data['Angle2_row'] # Retrieve angle row from child data

                        # Calculate power values
                        P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                        P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                        P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                        # Calculate time and energy values
                        T_l_vfly_value = H_value / V_lm_vfly_value
                        T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
                        E_ml_har_value = P_m_har_value * T_m_har_value # Corrected:


                        h_kml_down_value_compute=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
                        h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                        R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                        if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                            R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                        T_ml_down_value=D_m_current/R_ml_down_value
                        E_ml_down_value = P_m_down_value * T_ml_down_value
                        T_km_com_value = D_km / f_km_value
                        h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

                        R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                        T_km_up_value=D_m_current/R_kml_up_value # equation number 5
                        T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                        P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                        P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                        P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                        E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                        h_kml_har_value_compute=h_kml_down(Angle2_row,f_l_m_row,f_l_km_row) # Corrected index to Angle2_row
                        E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) # Corrected function call for E_kml_har
                        E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                        E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)


                        # Calculate fitness
                        fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                        current_data = {
                            'P_m_down_value': P_m_down_value,
                            'P_m_har_value': P_m_har_value,
                            'T_m_har_value': T_m_har_value,
                            'f_km_value': f_km_value,
                            'V_lm_vfly_value': V_lm_vfly_value,
                            'V_lm_hfly_value': V_lm_hfly_value,
                            'P_km_up_value':P_km_up_value,
                            'Angle1_row':Angle1_row,
                            'Angle_row':Angle_row,
                            'Angle2_row': Angle2_row, # Carry forward original index
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    }
                        if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                            return fitness_value, current_data
                        else:
                            return  float('inf'),{} # Return empty dict instead of float('inf') for data

                    child_fitness, child_data1 = compute_fitness(child_data)
                    child_population.append({'fitness': child_fitness, 'data': child_data1})

                # Create new population
                new_population = population + child_population
                new_population = sorted(new_population, key=lambda x: x['fitness'])
                population = new_population[:population_size]
                generations_data.append(population[0].copy())
                # print(population[0])

            best_individual_pair = population[0].copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'GA'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': population[0]['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })
            # print(f"Best Fitness for BS {l}, UAV {k}: {population[0]['fitness']:.4f}")

        # Find best individual for current BS across all UAVs
        best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])
        # print(f"Best Fitness for BS {l} across all UAVs: {best_individual_for_bs['fitness']:.4f}")


    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) ---")
    best_pair_for_plot = None
    min_fitness_for_plot = float('inf')

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f" UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f" Best Individual:")
        print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        unique_indices_to_print = assignment['unique_row_indices'] # Retrieve unique_row_indices
        for key, value in best_ind['data'].items():
            if isinstance(value, pd.Series):
                print(f"  {key}: Series: \n{value.iloc[unique_indices_to_print]}") # Print sliced Series
            elif isinstance(value, list): # Handle list type values explicitly
                print(f"  {key}: {value}") # print list directly without formatting
            else:
                print(f"  {key}: {value:.4f}") # Format scalar values

        print("-" * 20)
        # Changed line: Use the last generation's fitness
        last_generation_fitness = assignment['generation_fitness'][-1] # Access the last element
        sum_fitness_current_p_max += last_generation_fitness # Sum last generation fitness values

        if assignment['best_individual']['fitness'] < min_fitness_for_plot: # keep track of overall best fitness for plotting purposes - not related to sum_fitness_current_p_max
            min_fitness_for_plot = assignment['best_individual']['fitness']
            best_pair_for_plot = assignment

    return sum_fitness_current_p_max



fitness_sums_GA_IRS_RA= [] # Store sum of fitness values for each p_max

def GA_IRS_RA(D_m_current): # Define function to process each p_max value
    print(f"calculaiton for D_m_current for GA",D_m_current)
    all_best_combinations = []
    all_best_individuals = []
    sum_fitness_current_p_max = 0 # Initialize sum of fitness for current p_max

    # Main Genetic Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        # Select unique row indices for the current BS

        index_list = list(range(num_rows_data_files)) # Create a list of all indices
        random.shuffle(index_list)
        unique_row_indices = index_list[:num_irs_ele]
        # Create dataframes with uniquely selected rows for the current BS
        # unique_row_indices = range(0,50)
        h_l_km_df_bs = h_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        # Corrected line: Ensure indices are within bounds of P_km_up
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)


        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}
            population = []
            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value=0
            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)

            # Initialize population
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            T_l_vfly_value = H_value / V_lm_vfly_value
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)

            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)): # Using length of valid_indices
                f_km_value = f_km_bs[random.randint(0,population_size)] # Use BS-specific f_km
                P_km_up_value = P_km_up_bs[random.randint(0,population_size)] # Use BS-specific P_km_up

                Angle_row = Angle_df.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific Angle_df
                h_l_m_row = h_l_m_df.iloc[k, :] # Use BS-specific h_l_m_df
                h_l_km_row = h_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific h_l_km_df
                Angle1_row = Angle_UP_df.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific Angle_UP_df
                g_l_m_row = g_l_m_df.iloc[k, :] # Use BS-specific g_l_m_df
                g_l_km_row = g_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific g_l_km_df
                Angle2_row = Angle_har_df.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific Angle_har_df
                f_l_m_row = f_l_m_df.iloc[k, :] # Use BS-specific f_l_m_df
                f_l_km_row = f_l_km_df_bs.iloc[random.randint(0,num_irs_ele), :] # Use BS-specific f_l_km_df

                # Calculate power values


                # Calculate time and energy values
                # Corrected: D_l_hfly / V_lm_hfly
                E_ml_har_value = P_m_har_value * T_m_har_value
                h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Pass Series
                h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                T_ml_down_value=D_m_current/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=D_m_current/R_kml_up_value # equation number 5
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

                # Calculate fitness
                result_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

                # Store initial population data
                population.append({
                    'fitness': result_fitness,
                    'data':  {
                        'P_m_down_value': P_m_down_value,
                        'P_m_har_value': P_m_har_value,
                        'T_m_har_value': T_m_har_value,
                        'f_km_value': f_km_value,
                        'V_lm_vfly_value': V_lm_vfly_value,
                        'V_lm_hfly_value': V_lm_hfly_value,
                        'P_km_up_value':P_km_up_value,
                        'Angle1_row':Angle1_row,
                        'Angle_row':Angle_row,
                        'Angle2_row': Angle2_row,
                        }
                    })

            generations_data = []
            for j in range(num_generation):
                child_population = []
                # Corrected loop range to use valid_indices length
                for x in range(0, len(valid_indices), 2): # Loop through population with step of 2
                    if x + 1 >= len(valid_indices): # Check if i+1 is within bounds, if not break to avoid error in accessing population[i+1]
                        break
                    # Crossover
                    ranodmpopulation=[]
                    for i in range(10):
                        ranodmpopulation.append(random.choice(population))
                    ranodmpopulation = sorted(ranodmpopulation, key=lambda x: x['fitness'])
                    parent1 = ranodmpopulation[0]
                    parent2 = ranodmpopulation[1]
                    child_data = {}
                    for key in parent1['data']:
                        if key in numerical_keys_for_crossover:
                            child_data[key] = parent1['data'][key] # Keep parent1's numerical value
                            if key in ['Angle1_row','Angle_row','Angle2_row']: # Ensure angles are also Series after crossover
                                child_data[key] = pd.Series([random.uniform(1, 180) for _ in range(len(Angle_df.columns))], index=Angle_df.columns)
                            else:
                                child_data[key] = float(parent1['data'][key]) * 0.6 + float(parent2['data'][key]) * (1 - 0.6)
                        else:
                            child_data[key] = float(parent1['data'][key]) * 0.6 + float(parent2['data'][key]) * (1 - 0.6)

                    u = np.random.uniform(0, 1, 1)[0]
                    P_mutation = 0.5
                    if u < P_mutation:
                        for key in numerical_keys_for_crossover: # Apply mutation only to numerical keys
                            if key in ['Angle1_row','Angle_row','Angle2_row']: # Handle Angle Series
                                # child_data[key] = pd.Series(index=Angle_df.columns, dtype='float64') # Initialize empty Series for child
                                for col in Angle_df.columns: # Iterate through each column (angle direction)
                                    child_data[key][col] += random.normal(loc=0, scale=1, size=(1))[0]
                            else:
                                child_data[key] += random.normal(loc=0, scale=1, size=(1))[0] # Reduced scale for smaller perturbations in HC

                    # Compute child fitness
                    def compute_fitness(data):
                        P_m_down_value = data['P_m_down_value']
                        P_m_har_value = data['P_m_har_value']
                        T_m_har_value = data['T_m_har_value']
                        f_km_value = data['f_km_value']
                        V_lm_vfly_value = data['V_lm_vfly_value']
                        V_lm_hfly_value = data['V_lm_hfly_value']
                        P_km_up_value=data['P_km_up_value']
                        Angle_row = data['Angle_row'] # Retrieve angle row from child data
                        Angle1_row = data['Angle1_row'] # Retrieve angle row from child data
                        Angle2_row = data['Angle2_row'] # Retrieve angle row from child data

                        # Calculate power values
                        P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                        P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                        P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                        # Calculate time and energy values
                        T_l_vfly_value = H_value / V_lm_vfly_value
                        T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
                        E_ml_har_value = P_m_har_value * T_m_har_value # Corrected:


                        h_kml_down_value_compute=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Using original Angle_row, h_l_m_row, h_l_km_row for child as well - might need to be based on child data if angles are also part of optimization
                        h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                        R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                        if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                            R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                        T_ml_down_value=D_m_current/R_ml_down_value
                        E_ml_down_value = P_m_down_value * T_ml_down_value
                        T_km_com_value = D_km / f_km_value
                        h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

                        R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                        T_km_up_value=D_m_current/R_kml_up_value # equation number 5
                        T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                        P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                        P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                        P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                        E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                        h_kml_har_value_compute=h_kml_down(Angle2_row,f_l_m_row,f_l_km_row) # Corrected index to Angle2_row
                        E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) # Corrected function call for E_kml_har
                        E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                        E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)


                        # Calculate fitness
                        fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                        current_data = {
                            'P_m_down_value': P_m_down_value,
                            'P_m_har_value': P_m_har_value,
                            'T_m_har_value': T_m_har_value,
                            'f_km_value': f_km_value,
                            'V_lm_vfly_value': V_lm_vfly_value,
                            'V_lm_hfly_value': V_lm_hfly_value,
                            'P_km_up_value':P_km_up_value,
                            'Angle1_row':Angle1_row,
                            'Angle_row':Angle_row,
                            'Angle2_row': Angle2_row, # Carry forward original index
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    }
                        if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                            return fitness_value, current_data
                        else:
                            return  float('inf'),{} # Return empty dict instead of float('inf') for data

                    child_fitness, child_data1 = compute_fitness(child_data)
                    child_population.append({'fitness': child_fitness, 'data': child_data1})

                # Create new population
                new_population = population + child_population
                new_population = sorted(new_population, key=lambda x: x['fitness'])
                population = new_population[:population_size]
                generations_data.append(population[0].copy())
                # print(population[0])

            best_individual_pair = population[0].copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'GA'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': population[0]['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })
            # print(f"Best Fitness for BS {l}, UAV {k}: {population[0]['fitness']:.4f}")

        # Find best individual for current BS across all UAVs
        best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])
        # print(f"Best Fitness for BS {l} across all UAVs: {best_individual_for_bs['fitness']:.4f}")


    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) ---")
    best_pair_for_plot = None
    min_fitness_for_plot = float('inf')

    for assignment in best_assignments:
        print(f"\nBest Assignment for BS {assignment['bs_index']}:")
        print(f" UAV Index: {assignment['uav_index']}")
        best_ind = assignment['best_individual']
        print(f" Best Individual:")
        print(f"  Generation: {best_ind['generation']}, Type: {best_ind['type']}")
        print(f"  Fitness: {best_ind['fitness']:.4f}") # Print current best fitness only
        unique_indices_to_print = assignment['unique_row_indices'] # Retrieve unique_row_indices
        for key, value in best_ind['data'].items():
            if isinstance(value, pd.Series):
                print(f"  {key}: Series: \n{value.iloc[unique_indices_to_print]}") # Print sliced Series
            elif isinstance(value, list): # Handle list type values explicitly
                print(f"  {key}: {value}") # print list directly without formatting
            else:
                print(f"  {key}: {value:.4f}") # Format scalar values

        print("-" * 20)
        # Changed line: Use the last generation's fitness
        last_generation_fitness = assignment['generation_fitness'][-1] # Access the last element
        sum_fitness_current_p_max += last_generation_fitness # Sum last generation fitness values

        if assignment['best_individual']['fitness'] < min_fitness_for_plot: # keep track of overall best fitness for plotting purposes - not related to sum_fitness_current_p_max
            min_fitness_for_plot = assignment['best_individual']['fitness']
            best_pair_for_plot = assignment

    return sum_fitness_current_p_max



fitness_sums_HC_IRS = [] # Store sum of fitness values for each p_max


# Define a function to process each p_max value
def HC_IRS(D_m_current):
    print(f"calculaiton for D_m_current for HC",D_m_current)
    all_best_combinations = []
    all_best_individuals = []
    sum_fitness_current_p_max = 0 # Initialize sum of fitness for current p_max

    # Main Hill Climbing Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        # Select unique row indices for the current BS
        index_list = list(range(num_rows_data_files)) # Create a list of all indices
        random.shuffle(index_list)
        unique_row_indices = index_list[:num_irs_ele] # use population size to pick initial indices
        # Create dataframes with uniquely selected rows for the current BS
        # unique_row_indices = range(0,50)
        h_l_km_df_bs = h_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        # Corrected line: Ensure indices are within bounds of P_km_up
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)


        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}

            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value=0
            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)

            # Initialize initial solution for Hill Climbing - using first from 'population initialization' of GA
            initial_solution_data = {}
            i=0 # Using first index for initialization
            f_km_value = f_km_bs[i] # Use BS-specific f_km
            P_km_up_value = P_km_up_bs[i] # Use BS-specific P_km_up

            Angle_row = Angle_df.iloc[i, :] # Use BS-specific Angle_df
            h_l_m_row = h_l_m_df.iloc[k, :] # Use BS-specific h_l_m_df
            h_l_km_row = h_l_km_df_bs.iloc[i, :] # Use BS-specific h_l_km_df
            Angle1_row = Angle_UP_df.iloc[i, :] # Use BS-specific Angle_UP_df
            g_l_m_row = g_l_m_df.iloc[k, :] # Use BS-specific g_l_m_df
            g_l_km_row = g_l_km_df_bs.iloc[i, :] # Use BS-specific g_l_km_df
            Angle2_row = Angle_har_df.iloc[i, :] # Use BS-specific Angle_har_df
            f_l_m_row = f_l_m_df.iloc[k, :] # Use BS-specific f_l_m_df
            f_l_km_row = f_l_km_df_bs.iloc[i, :] # Use BS-specific f_l_km_df



            # Calculate power values
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

            # Calculate time and energy values
            T_l_vfly_value = H_value / V_lm_vfly_value
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
            E_ml_har_value = P_m_har_value * T_m_har_value
            h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Pass Series
            h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
            T_ml_down_value=D_m_current/R_ml_down_value
            E_ml_down_value = P_m_down_value * T_ml_down_value
            T_km_com_value = D_km / f_km_value
            h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

            R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
            T_km_up_value=D_m_current/R_kml_up_value # equation number 5
            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

            # Calculate fitness for initial solution
            initial_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

            current_solution = {
                'fitness': initial_fitness,
                'data': {
                    'P_m_down_value': P_m_down_value,
                    'P_m_har_value': P_m_har_value,
                    'T_m_har_value': T_m_har_value,
                    'f_km_value': f_km_value,
                    'V_lm_vfly_value': V_lm_vfly_value,
                    'V_lm_hfly_value': V_lm_hfly_value,
                    'P_km_up_value':P_km_up_value,
                    'Angle1_row':Angle1_row,
                    'Angle_row':Angle_row,
                    'Angle2_row': Angle2_row,
                }
            }
            best_individual = current_solution
            best_fitness = initial_fitness


            generations_data = []
            for j in range(num_generation): # Hill Climbing iterations
                # Generate neighbor solution by perturbing the current solution
                neighbor_solution_data = current_solution['data'].copy() # Correct: Copy current solution data


                for i in range(4): # You can keep this loop if it's intended for repeated perturbations per generation
                    for key in numerical_keys_for_crossover:
                        # Apply mutation only to numerical keys
                        if key in ['Angle1_row','Angle_row','Angle2_row']: # Handle Angle Series
                            # --- REMOVED: neighbor_solution_data[key] = pd.Series(index=Angle_df.columns, dtype='float64') --- # PROBLEM LINE REMOVED
                            for col in Angle_df.columns: # Iterate through each column (angle direction)
                                neighbor_solution_data[key][col] += random.normal(loc=0, scale=1, size=(1))[0] # Perturb EXISTING value
                                if neighbor_solution_data[key][col] < 0: # Check if the RESULTING angle is negative
                                    neighbor_solution_data[key][col] = abs(neighbor_solution_data[key][col]) # Take abs value of the RESULT

                        else:
                            neighbor_solution_data[key] += random.normal(loc=0, scale=1, size=(1))[0] # Reduced scale for smaller perturbations in HC


                        # Compute neighbor fitness (rest of your code remains the same)
                        def compute_fitness(data): # Define compute_fitness WITHIN the generation loop - scope is fine here
                            P_m_down_value = data['P_m_down_value']
                            P_m_har_value = data['P_m_har_value']
                            T_m_har_value = data['T_m_har_value']
                            f_km_value = data['f_km_value']
                            V_lm_vfly_value = data['V_lm_vfly_value']
                            V_lm_hfly_value = data['V_lm_hfly_value']
                            P_km_up_value=data['P_km_up_value']
                            Angle_row = data['Angle_row'] # Retrieve angle row from neighbor data
                            Angle1_row = data['Angle1_row'] # Retrieve angle row from neighbor data
                            Angle2_row = data['Angle2_row'] # Retrieve angle row from neighbor data

                            # Calculate power values
                            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                            # Calculate time and energy values
                            T_l_vfly_value = H_value / V_lm_vfly_value
                            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
                            E_ml_har_value = P_m_har_value * T_m_har_value # Corrected:

                            h_kml_down_value_compute=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Using original Angle_row, h_l_m_row, h_l_km_row for neighbor as well - might need to be based on neighbor data if angles are also part of optimization
                            h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                            if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                                R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                            T_ml_down_value=D_m_current/R_ml_down_value
                            E_ml_down_value = P_m_down_value * T_ml_down_value
                            T_km_com_value = D_km / f_km_value

                            h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different
                            R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                            T_km_up_value=D_m_current/R_kml_up_value # equation number 5

                            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                            h_kml_har_value_compute=h_kml_down(Angle2_row,f_l_m_row,f_l_km_row) # Corrected index to Angle2_row
                            E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) # Corrected function call for E_kml_har
                            E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                            E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)


                            # Calculate fitness
                            fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                            current_data = {
                                'P_m_down_value': P_m_down_value,
                                'P_m_har_value': P_m_har_value,
                                'T_m_har_value': T_m_har_value,
                                'f_km_value': f_km_value,
                                'V_lm_vfly_value': V_lm_vfly_value,
                                'V_lm_hfly_value': V_lm_hfly_value,
                                'P_km_up_value':P_km_up_value,
                                'Angle1_row':Angle1_row,
                                'Angle_row':Angle_row,
                                'Angle2_row': Angle2_row, # Carry forward original index
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    }
                            if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                                return fitness_value, current_data
                            else:
                                return  float('inf'),{} # Return empty dict instead of float('inf') for data


                        neighbor_fitness, neighbor_data1 = compute_fitness(neighbor_solution_data)


                        # Decide whether to accept the neighbor
                        if neighbor_fitness < current_solution['fitness']: # Assuming minimization
                            current_solution = {'fitness': neighbor_fitness, 'data': neighbor_data1}


                generations_data.append(current_solution.copy()) # Append the *updated* current_solution

            best_individual_pair = current_solution.copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'HC'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': current_solution['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })


        # Find best individual for current BS across all UAVs
        best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])
        # print(f"Best Fitness for BS {l} across all UAVs: {best_individual_for_bs['fitness']:.4f}")


    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) using Hill Climbing ---")
    best_pair_for_plot = None
    min_fitness_for_plot = float('inf')

    sum_fitness_current_p_max = 0 # Sum of best fitness for current p_max
    for assignment in best_assignments:
        best_ind = assignment['best_individual']
        # Changed line: Access last generation fitness and sum it
        last_generation_fitness = assignment['generation_fitness'][-1]
        sum_fitness_current_p_max += last_generation_fitness

    return sum_fitness_current_p_max # Return sum of fitness for this p_max



fitness_sums_HC_IRS_RA = [] # Store sum of fitness values for each p_max


# Define a function to process each p_max value
def HC_IRS_RA(D_m_current):
    print(f"calculaiton for D_m_current for HC",D_m_current)
    all_best_combinations = []
    all_best_individuals = []
    sum_fitness_current_p_max = 0 # Initialize sum of fitness for current p_max

    # Main Hill Climbing Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        # Select unique row indices for the current BS
        index_list = list(range(num_rows_data_files)) # Create a list of all indices
        random.shuffle(index_list)
        unique_row_indices = index_list[:num_irs_ele] # use population size to pick initial indices
        # Create dataframes with uniquely selected rows for the current BS
        # unique_row_indices = range(0,50)
        h_l_km_df_bs = h_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True)
        # Corrected line: Ensure indices are within bounds of P_km_up
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)]
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)


        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}

            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]

            Sub_value=0
            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)

            # Initialize initial solution for Hill Climbing - using first from 'population initialization' of GA
            initial_solution_data = {}
            i=0 # Using first index for initialization
            f_km_value = f_km_bs[i] # Use BS-specific f_km
            P_km_up_value = P_km_up_bs[i] # Use BS-specific P_km_up

            Angle_row = Angle_df.iloc[i, :] # Use BS-specific Angle_df
            h_l_m_row = h_l_m_df.iloc[k, :] # Use BS-specific h_l_m_df
            h_l_km_row = h_l_km_df_bs.iloc[i, :] # Use BS-specific h_l_km_df
            Angle1_row = Angle_UP_df.iloc[i, :] # Use BS-specific Angle_UP_df
            g_l_m_row = g_l_m_df.iloc[k, :] # Use BS-specific g_l_m_df
            g_l_km_row = g_l_km_df_bs.iloc[i, :] # Use BS-specific g_l_km_df
            Angle2_row = Angle_har_df.iloc[i, :] # Use BS-specific Angle_har_df
            f_l_m_row = f_l_m_df.iloc[k, :] # Use BS-specific f_l_m_df
            f_l_km_row = f_l_km_df_bs.iloc[i, :] # Use BS-specific f_l_km_df



            # Calculate power values
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

            # Calculate time and energy values
            T_l_vfly_value = H_value / V_lm_vfly_value
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
            E_ml_har_value = P_m_har_value * T_m_har_value
            h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Pass Series
            h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
            T_ml_down_value=D_m_current/R_ml_down_value
            E_ml_down_value = P_m_down_value * T_ml_down_value
            T_km_com_value = D_km / f_km_value
            h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

            R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
            T_km_up_value=D_m_current/R_kml_up_value # equation number 5
            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

            # Calculate fitness for initial solution
            initial_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

            current_solution = {
                'fitness': initial_fitness,
                'data': {
                    'P_m_down_value': P_m_down_value,
                    'P_m_har_value': P_m_har_value,
                    'T_m_har_value': T_m_har_value,
                    'f_km_value': f_km_value,
                    'V_lm_vfly_value': V_lm_vfly_value,
                    'V_lm_hfly_value': V_lm_hfly_value,
                    'P_km_up_value':P_km_up_value,
                    'Angle1_row':Angle1_row,
                    'Angle_row':Angle_row,
                    'Angle2_row': Angle2_row,
                }
            }
            best_individual = current_solution
            best_fitness = initial_fitness


            generations_data = []
            for j in range(num_generation): # Hill Climbing iterations
                # Generate neighbor solution by perturbing the current solution
                neighbor_solution_data = current_solution['data'].copy() # Correct: Copy current solution data

                for i in range(4):
                    for key in numerical_keys_for_crossover:
                        if key in ['Angle1_row','Angle_row','Angle2_row']: # Ensure angles are also Series
                            # Select random angles for Angle Series instead of perturbing
                            neighbor_solution_data[key] = pd.Series([random.uniform(1, 180) for _ in range(len(Angle_df.columns))], index=Angle_df.columns)
                        else:
                            neighbor_solution_data[key] += random.normal(loc=0, scale=1, size=(1))[0] # Perturb other numerical values


                        # Compute neighbor fitness (rest of your code remains the same)
                        def compute_fitness(data): # Define compute_fitness WITHIN the generation loop - scope is fine here
                            P_m_down_value = data['P_m_down_value']
                            P_m_har_value = data['P_m_har_value']
                            T_m_har_value = data['T_m_har_value']
                            f_km_value = data['f_km_value']
                            V_lm_vfly_value = data['V_lm_vfly_value']
                            V_lm_hfly_value = data['V_lm_hfly_value']
                            P_km_up_value=data['P_km_up_value']
                            Angle_row = data['Angle_row'] # Retrieve angle row from neighbor data
                            Angle1_row = data['Angle1_row'] # Retrieve angle row from neighbor data
                            Angle2_row = data['Angle2_row'] # Retrieve angle row from neighbor data

                            # Calculate power values
                            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                            # Calculate time and energy values
                            T_l_vfly_value = H_value / V_lm_vfly_value
                            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
                            E_ml_har_value = P_m_har_value * T_m_har_value # Corrected:

                            h_kml_down_value_compute=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Using original Angle_row, h_l_m_row, h_l_km_row for neighbor as well - might need to be based on neighbor data if angles are also part of optimization
                            h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                            if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                                R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                            T_ml_down_value=D_m_current/R_ml_down_value
                            E_ml_down_value = P_m_down_value * T_ml_down_value
                            T_km_com_value = D_km / f_km_value

                            h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different
                            R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                            T_km_up_value=D_m_current/R_kml_up_value # equation number 5

                            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                            h_kml_har_value_compute=h_kml_down(Angle2_row,f_l_m_row,f_l_km_row) # Corrected index to Angle2_row
                            E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) # Corrected function call for E_kml_har
                            E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                            E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)


                            # Calculate fitness
                            fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                            current_data = {
                                'P_m_down_value': P_m_down_value,
                                'P_m_har_value': P_m_har_value,
                                'T_m_har_value': T_m_har_value,
                                'f_km_value': f_km_value,
                                'V_lm_vfly_value': V_lm_vfly_value,
                                'V_lm_hfly_value': V_lm_hfly_value,
                                'P_km_up_value':P_km_up_value,
                                'Angle1_row':Angle1_row,
                                'Angle_row':Angle_row,
                                'Angle2_row': Angle2_row, # Carry forward original index
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    }
                            if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                                return fitness_value, current_data
                            else:
                                return  float('inf'),{} # Return empty dict instead of float('inf') for data


                        neighbor_fitness, neighbor_data1 = compute_fitness(neighbor_solution_data)


                        # Decide whether to accept the neighbor
                        if neighbor_fitness < current_solution['fitness']: # Assuming minimization
                            current_solution = {'fitness': neighbor_fitness, 'data': neighbor_data1}


                generations_data.append(current_solution.copy()) # Append the *updated* current_solution

            best_individual_pair = current_solution.copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'HC'
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': current_solution['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })


        # Find best individual for current BS across all UAVs
        best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])
        # print(f"Best Fitness for BS {l} across all UAVs: {best_individual_for_bs['fitness']:.4f}")


    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) using Hill Climbing ---")
    best_pair_for_plot = None
    min_fitness_for_plot = float('inf')

    sum_fitness_current_p_max = 0 # Sum of best fitness for current p_max
    for assignment in best_assignments:
        best_ind = assignment['best_individual']
        # Changed line: Access last generation fitness and sum it
        last_generation_fitness = assignment['generation_fitness'][-1]
        sum_fitness_current_p_max += last_generation_fitness

    return sum_fitness_current_p_max # Return sum of fitness for this p_max



fitness_sums_RS = [] # Store sum of fitness values for each p_max

# Define a function to process each p_max value
def RS(D_m_current):
    print(f"calculaiton for D_m_current for RS",D_m_current) # corrected print statement
    all_best_combinations = []
    all_best_individuals = []
    sum_fitness_current_p_max = 0 # Initialize sum of fitness for current p_max

    # Main Hill Climbing Algorithm Loop
    for l in range(num_bs):
        all_best_individuals_bs = []
        P_m_har_value = P_m_har.values[l]
        T_m_har_value = T_m_har.values[l]
        P_m_down_value = P_m_down.values[l]

        # Select unique row indices for the current BS
        index_list = list(range(num_rows_data_files)) # Create a list of all indices
        random.shuffle(index_list)
        unique_row_indices = index_list[:num_irs_ele] # use population size to pick initial indices
        # Create dataframes with uniquely selected rows for the current BS
        # unique_row_indices = range(0,50) # Use random.sample to select unique indices
        h_l_km_df_bs = h_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        g_l_km_df_bs = g_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_l_km_df_bs = f_l_km_df.iloc[unique_row_indices, :].reset_index(drop=True)
        f_km_bs = f_km[unique_row_indices[0:population_size]].reset_index(drop=True) # corrected indexing
        # Corrected line: Ensure indices are within bounds of P_km_up
        valid_indices = [i for i in unique_row_indices[0:population_size] if i < len(P_km_up)] # corrected indexing
        P_km_up_bs = P_km_up.iloc[valid_indices].reset_index(drop=True)


        for k in range(num_uav_irs):
            best_fitness = float('inf')
            best_individual = {}

            V_lm_vfly_value = V_lm_vfly.values[k]
            V_lm_hfly_value = V_lm_hfly.values[k]
            Sub_value=0
            # Corrected loop range to use valid_indices length
            for i in range(len(valid_indices)): # Using length of valid_indices
                h_il_up_value=h_kml_down(Angle_UP_df.iloc[i, :],g_l_m_df.iloc[k, :],g_l_km_df_bs.iloc[i, :]) # Pass Series, corrected index to i
                Sub_value+=sub(P_km_up_bs[i],h_il_up_value)

            # Initialize initial solution for Hill Climbing - using first from 'population initialization' of GA
            initial_solution_data = {}
            i=0 # Using first index for initialization
            f_km_value = f_km_bs[i] # Use BS-specific f_km
            P_km_up_value = P_km_up_bs[i] # Use BS-specific P_km_up

            Angle_row = Angle_df.iloc[i, :] # Use BS-specific Angle_df
            h_l_m_row = h_l_m_df.iloc[k, :] # Use BS-specific h_l_m_df
            h_l_km_row = h_l_km_df_bs.iloc[i, :] # Use BS-specific h_l_km_df
            Angle1_row = Angle_UP_df.iloc[i, :] # Use BS-specific Angle_UP_df
            g_l_m_row = g_l_m_df.iloc[k, :] # Use BS-specific g_l_m_df
            g_l_km_row = g_l_km_df_bs.iloc[i, :] # Use BS-specific g_l_km_df
            Angle2_row = Angle_har_df.iloc[i, :] # Use BS-specific Angle_har_df
            f_l_m_row = f_l_m_df.iloc[k, :] # Use BS-specific f_l_m_df
            f_l_km_row = f_l_km_df_bs.iloc[i, :] # Use BS-specific f_l_km_df



            # Calculate power values
            P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
            P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
            P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

            # Calculate time and energy values
            T_l_vfly_value = H_value / V_lm_vfly_value
            T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
            E_ml_har_value = P_m_har_value * T_m_har_value
            h_kml_down_value=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Pass Series
            h_ml_worst_value=h_ml_worst(h_kml_down_value,sigma_km)
            R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
            T_ml_down_value=D_m_current/R_ml_down_value
            E_ml_down_value = P_m_down_value * T_ml_down_value
            T_km_com_value = D_km / f_km_value
            h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different

            R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
            T_km_up_value=D_m_current/R_kml_up_value # equation number 5
            T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
            P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
            P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
            P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
            E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)

            # Calculate fitness for initial solution
            initial_fitness = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)

            current_solution = {
                'fitness': initial_fitness,
                'data': {
                    'P_m_down_value': P_m_down_value,
                    'P_m_har_value': P_m_har_value,
                    'T_m_har_value': T_m_har_value,
                    'f_km_value': f_km_value,
                    'V_lm_vfly_value': V_lm_vfly_value,
                    'V_lm_hfly_value': V_lm_hfly_value,
                    'P_km_up_value':P_km_up_value,
                    'Angle1_row':Angle1_row,
                    'Angle_row':Angle_row,
                    'Angle2_row': Angle2_row,
                }
            }
            best_individual = current_solution
            best_fitness = initial_fitness


            generations_data = []
            # Compute neighbor fitness function (defined outside loops for efficiency)
            def compute_fitness(data):
                P_m_down_value = data['P_m_down_value']
                P_m_har_value = data['P_m_har_value']
                T_m_har_value = data['T_m_har_value']
                f_km_value = data['f_km_value']
                V_lm_vfly_value = data['V_lm_vfly_value']
                V_lm_hfly_value = data['V_lm_hfly_value']
                P_km_up_value = data['P_km_up_value']
                Angle_row = data['Angle_row'] # Retrieve angle row from neighbor data
                Angle1_row = data['Angle1_row'] # Retrieve angle row from neighbor data
                Angle2_row = data['Angle2_row'] # Retrieve angle row from neighbor data

                # Calculate power values
                P_lm_blade_value = P_lm_blade(Nr, p_l_b, V_tip, V_lm_hfly_value)
                P_lm_fuselage_value = P_lm_fuselage(Cd, Af, Bh, V_lm_hfly_value)
                P_lm_induced_value = P_lm_induced(Nr, Bh, Ar, Wl_value, V_lm_vfly_value)

                # Calculate time and energy values
                T_l_vfly_value = H_value / V_lm_vfly_value
                T_l_hfly_value = D_l_hfly_value / V_lm_hfly_value # Corrected: D_l_hfly / V_lm_hfly
                E_ml_har_value = P_m_har_value * T_m_har_value # Corrected:

                h_kml_down_value_compute=h_kml_down(Angle_row,h_l_m_row,h_l_km_row) # Using original Angle_row, h_l_m_row, h_l_km_row for neighbor as well - might need to be based on neighbor data if angles are also part of optimization
                h_ml_worst_value=h_ml_worst(h_kml_down_value_compute,sigma_km)
                R_ml_down_value=R_ml_down(B,P_m_down_value,h_ml_worst_value)
                if R_ml_down_value <= 0: # check if R_ml_down_value is zero or negative
                    R_ml_down_value = 1e-9 # Assign a small positive value to avoid division by zero
                T_ml_down_value=D_m_current/R_ml_down_value
                E_ml_down_value = P_m_down_value * T_ml_down_value
                T_km_com_value = D_km / f_km_value
                h_kml_up_value=h_kml_down(Angle1_row,g_l_m_row,g_l_km_row) # Pass Series, using same function, might need different one if logic is different
                R_kml_up_value=R_kml_up(B,P_km_up_value,h_kml_up_value,Sub_value,sigma_km)
                T_km_up_value=D_m_current/R_kml_up_value # equation number 5
                T_lm_hov_value = T_lm_hov(T_km_com_value, T_km_up_value, T_ml_down_value)
                P_l_hov_value = P_l_hov(Wl_value, p_l_b, Nr, Ar, Bh)
                P_l_vfly_value = P_l_vfly(Wl_value, V_lm_vfly_value, p_l_b, Nr, Ar, Bh)
                P_lm_hfly_value = P_lm_hfly(P_lm_blade_value, P_lm_fuselage_value, P_lm_induced_value)
                E_ml_UAV_value = E_ml_UAV(P_l_vfly_value, T_l_vfly_value, P_lm_hfly_value, T_l_hfly_value, P_l_hov_value, T_lm_hov_value)
                h_kml_har_value_compute=h_kml_down(Angle2_row,f_l_m_row,f_l_km_row) # Corrected index to Angle2_row

                E_kml_har_value=E_kml_har(P_m_har_value,T_m_har_value,h_kml_har_value_compute) # Corrected function call for E_kml_har
                E_kml_com_value = E_km_com(f_km_value, T_km_com_value)
                E_kml_up_value=E_kml_up(P_km_up_value,T_km_up_value)


                # Calculate fitness
                fitness_value = Fitness(E_ml_har_value, E_ml_down_value, E_ml_UAV_value)
                current_data = {
                    'P_m_down_value': P_m_down_value,
                    'P_m_har_value': P_m_har_value,
                    'T_m_har_value': T_m_har_value,
                    'f_km_value': f_km_value,
                    'V_lm_vfly_value': V_lm_vfly_value,
                    'V_lm_hfly_value': V_lm_hfly_value,
                    'P_km_up_value':P_km_up_value,
                    'Angle1_row':Angle1_row,
                    'Angle_row':Angle_row,
                    'Angle2_row': Angle2_row, # Carry forward original index
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                }
                if V_lm_hfly_value>0 and T_m_har_value>0 and T_ml_down_value>0 and T_km_up_value>0 and P_m_har_value<=p_max and P_m_down_value<=p_max and P_km_up_value<=p_km_max and (T_km_com_value+T_km_up_value+T_ml_down_value)<=T_m and f_km_value>0  and E_kml_har_value>=(E_kml_up_value+E_kml_com_value) and V_lm_vfly_value>0:
                    return fitness_value, current_data
                else:
                    return  float('inf'),{} # Return empty dict instead of float('inf') for data



            for j in range(num_generation): # Hill Climbing iterations
                # for i in range(10): # Generate 10 neighbor solutions per generation
                neighbor_solution_data = current_solution['data'].copy() # Start with current solution for each neighbor

                for key in numerical_keys_for_crossover: # Perturb each key for the current neighbor
                    if key in ['Angle1_row','Angle_row','Angle2_row']:
                        neighbor_solution_data[key] = pd.Series([random.uniform(1, 180) for _ in range(len(Angle_df.columns))], index=Angle_df.columns)
                    elif key in ['P_m_down_value', 'P_m_har_value', 'T_m_har_value','f_km_value']:
                        neighbor_solution_data[key] = random.uniform(0, 1) # Corrected: Assign directly to key
                    elif key in ['V_lm_vfly_value', 'V_lm_hfly_value']:
                        neighbor_solution_data[key] = random.uniform(0, 100) # Corrected: Assign directly to key
                    elif key in ['P_km_up_value']:
                        neighbor_solution_data[key] = random.uniform(0, 10) # Corrected: Assign directly to key
                    else:
                        neighbor_solution_data[key] = neighbor_solution_data[key] + np.random.normal(loc=0, scale=1, size=(1))[0] # Corrected line

                neighbor_fitness, neighbor_data1 = compute_fitness(neighbor_solution_data)

                # Decide whether to accept the neighbor
                if neighbor_fitness < current_solution['fitness']:
                    current_solution = {'fitness': neighbor_fitness, 'data': neighbor_data1}

                generations_data.append(current_solution.copy())

            best_individual_pair = current_solution.copy()
            best_individual_pair['generation'] = j + 1 # Use last j from loop, corrected index
            best_individual_pair['type'] = 'RS' # corrected type to RS
            best_individual_pair['bs_index'] = l
            best_individual_pair['uav_index'] = k
            all_best_individuals_bs.append(best_individual_pair)

            sum_fitness_current_p_max_uav = 0 # Initialize sum of fitness for current p_max and UAV
            all_best_combinations.append({
                'bs_index': l,
                'uav_index': k,
                'best_fitness': current_solution['fitness'],
                'best_individual': best_individual_pair,
                'generation_fitness': [gen['fitness'] for gen in generations_data],
                'unique_row_indices': unique_row_indices # Store unique_row_indices in best_combinations
            })
            sum_fitness_current_p_max_uav = current_solution['fitness'] # should return fitness value not object

        # Find best individual for current BS across all UAVs
        best_individual_for_bs = min(all_best_individuals_bs, key=lambda x: x['fitness'])
        # print(f"Best Fitness for BS {l} across all UAVs: {best_individual_for_bs['fitness']:.4f}")


    # Select the best unique Base station and UAV-IRS pair using Auction based method
    combination_lookup = {}
    for combination in all_best_combinations:
        if combination['bs_index'] not in combination_lookup:
            combination_lookup[combination['bs_index']] = {}
        combination_lookup[combination['bs_index']][combination['uav_index']] = combination

    # Auction-based assignment
    best_assignments = []
    unassigned_bs = list(range(num_bs))
    unassigned_uavs = list(range(num_uav_irs))

    while unassigned_bs and unassigned_uavs:
        best_combination_overall = None

        for l in unassigned_bs:
            best_fitness_for_bs = float('inf')
            best_combination_for_bs = None
            for k in unassigned_uavs:
                if l in combination_lookup and k in combination_lookup[l]:
                    combination = combination_lookup[l][k]
                    if combination['best_fitness'] < best_fitness_for_bs: # Use 'best_fitness' instead of 'best_individual']['fitness']
                        best_fitness_for_bs = combination['best_fitness']
                        best_combination_for_bs = combination

            if best_combination_for_bs:
                if best_combination_overall is None or best_combination_for_bs['best_fitness'] < best_combination_overall['best_fitness']: # Compare with current best overall
                    best_combination_overall = best_combination_for_bs

        if best_combination_overall:
            best_assignments.append(best_combination_overall)
            unassigned_bs.remove(best_combination_overall['bs_index'])
            unassigned_uavs.remove(best_combination_overall['uav_index'])

    # # Print and Plotting
    print(f"\n--- Best Unique UAV Assignments (Auction Based Method) using Random Search ---") # corrected print statement
    best_pair_for_plot = None
    min_fitness_for_plot = float('inf')

    sum_fitness_current_p_max = 0 # Sum of best fitness for current p_max
    for assignment in best_assignments:
        best_ind = assignment['best_individual']
        # Changed line: Access last generation fitness and sum it
        last_generation_fitness = assignment['generation_fitness'][-1]
        sum_fitness_current_p_max += last_generation_fitness # corrected to sum last generation fitness

    return sum_fitness_current_p_max # Return sum of fitness for this p_max.in this code i want sum_fitness_current_p_max this variable return the sum of fitness of each unique best pair of last generation for each D_m_current .update the code to do this


if __name__ == '__main__': # Add this to prevent issues in multiprocessing on Windows
    D_m_current_values = np.arange(0.1, 1.1, 0.1) # P_max values from 1 to 11
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        fitness_sums_GA_IRS = pool.map(GA_IRS, D_m_current_values)
        fitness_sums_GA_IRS_RA = pool.map(GA_IRS_RA, D_m_current_values)
        fitness_sums_HC_IRS = pool.map(HC_IRS, D_m_current_values)
        fitness_sums_HC_IRS_RA = pool.map(HC_IRS_RA, D_m_current_values)
        fitness_sums_RS = pool.map(RS, D_m_current_values)
        

    data_dict = {
        "D_m_value": list(range(1, len(fitness_sums_GA_IRS) + 1)) if fitness_sums_GA_IRS else [], # Assuming generation number starts from 1
        "fitness_sums_HC_IRS_RA": fitness_sums_HC_IRS_RA,
        "fitness_sums_HC_IRS": fitness_sums_HC_IRS,
        "fitness_sums_GA_IRS_RA": fitness_sums_GA_IRS_RA,
        "fitness_sums_GA_IRS": fitness_sums_GA_IRS,
        "fitness_sums_RS": fitness_sums_RS,
    }

    csv_file_path_pandas = "fitness_summary_Data Size.csv"

    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path_pandas, index=False) # index=False to prevent writing row indices to CSV


    plt.figure(figsize=(12, 7)) 
    D_m_current_values =np.arange(0.1, 1.1, 0.1)
    plt.rcParams["font.size"] = "20"
    plt.plot(D_m_current_values, fitness_sums_GA_IRS, marker='*', linestyle='dotted',label = "C2G-A_IRS")
    plt.plot(D_m_current_values, fitness_sums_GA_IRS_RA, marker='s', linestyle='dotted',label = "C2G-A_IRS_RA")
    plt.plot(D_m_current_values, fitness_sums_HC_IRS, marker='s', linestyle='-',label = "HC-IRS")
    plt.plot(D_m_current_values, fitness_sums_HC_IRS_RA, marker='*', linestyle='-',label = "HC-IRS_RA")
    plt.plot(D_m_current_values, fitness_sums_RS, marker='o', linestyle='dashdot',label = "RS")
    plt.xlabel('Data size (Dm)',size=20)
    plt.ylabel('Energy',size=22)
    plt.legend()
    plt.savefig("Energy vs Data size.pdf", format="pdf", bbox_inches="tight", dpi=800) # saved with different name
    plt.show()
