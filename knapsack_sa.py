'''
This script is a simulated annealing class that can be used to solve the
the 0/1 knapsack problem. It makes uses of a combination of techniques to get the
closest value to optimal. This techniques are greedy initialization and restart.

'''
# Libraries
import numpy as np
import time
import os

class SimulatedAnnealingKnapsack:
    def __init__(self, file_name, file_path, verbose=False):
        '''Reads input parameters and  creates global variables.'''

        self.file_path = file_path
        self.w_limit, self.total_items, self.item_values, self.item_weights = self.get_data()
        self.verbose = verbose
        self.file_name = file_name

        

    def get_data(self):
        """
        Reads input files and initialize parameters.

        Returns:
            w_limit (int): The maximum weight limit allowed.
            total_items (int): The total number of items in the file.
            item_values (np.array): An array of floats representing the values of the items.
            item_weights (np.array): An array of floats representing the weights of the items.
        """
        w_limit = None
        total_items = None
        item_values = []
        item_weights = []

        with open(self.file_path, 'r') as file:
            for i, line in enumerate(file):
                number1, number2 = line.split()
                number1, number2 = float(number1), float(number2)

                if i == 0:
                    total_items = int(number1)
                    w_limit = int(number2)
                else:
                    item_values.append(number1)
                    item_weights.append(number2)
        
        item_values = np.array(item_values, dtype=float)
        item_weights = np.array(item_weights, dtype=float)

        return w_limit, total_items, item_values, item_weights
    
    def get_runtime(self, start_time):
        """ Returns the elapsed time in seconds based on a start time"""

        curr_time = time.time()  # Current time
        elapsed_time = curr_time - start_time
        return elapsed_time

    
    def flip_random_element(self, state, max_flips):
        """
        Flip one or more random elements in the input state
        and returns the new state.
        """
        if max_flips > len(state):
            max_flips = len(state)
            if self.verbose:
                print('Max flips > Total items')
                print('Limiting max flips to total items...')

        flipped_indices = set()
        while len(flipped_indices) < max_flips:
            random_index = np.random.randint(len(state))
            state[random_index] = 1 - state[random_index]
            flipped_indices.add(random_index)
            if len(state) == len(flipped_indices):  # If all indices have been flipped
                break

        return state
    
    def adjust_max_flips(self, initial_max_flips, acceptance_rate, target_acceptance=0.44, min_flips=1):
        """Returns an adjusted the number of flips based on acceptance rate."""

        if self.verbose: print('target acceptance: ', target_acceptance)

        # Adjust based on acceptance rate
        if acceptance_rate > target_acceptance:
            acceptance_factor = 1.1  # Increase flips by 10% if acceptance is high
        else:
            acceptance_factor = 0.9  # Decrease flips by 10% if acceptance is low

    
        adjusted_flips = initial_max_flips * acceptance_factor
        adjusted_flips = max(min_flips, min(self.total_items, adjusted_flips))  # Ensure within bounds

        return int(adjusted_flips)
    
    def heating_temperature(self, curr_state, max_flips, acceptance_target=0.99, trials=100, multiplier=1.1):
        """
        Returns suitable initial temperature so that the acceptance rate is close to 1.
        This method comes from the Handbook for Metaheuristics Chapter 1.4.2.
        """
        if self.verbose:
            print('----------------------')
            print('Starting heating process...')


        T_start = 1.0  # Initial temperature guess
        accepted = 0

        curr_value = np.sum(curr_state * self.item_values)

        acceptance_rate = float('-inf')
        # Adjust T_start until acceptance rate is close to the target
        while acceptance_rate < acceptance_target:
            T_start *= multiplier
            accepted = 0
            for _ in range(trials):
                candidate_state = self.flip_random_element(np.copy(curr_state), max_flips)
                candidate_value = np.sum(candidate_state * self.item_values)

                exp_argument = (candidate_value - curr_value) / T_start
                exp_argument = np.clip(exp_argument, None, 700)  # Avoids overflowing
                acceptance_prob = np.exp(exp_argument)

                if candidate_value >= curr_value or np.random.rand() < acceptance_prob:
                    accepted += 1
            acceptance_rate = accepted / trials

            if self.verbose:
                print('acceptance rate: ', acceptance_rate)

        return T_start

    def generate_greedy_feasible_initial_state(self):
        """
        Constructs an initial state for a knapsack problem using a greedy approach based on
        value-to-weight ratios. Items are added to the state until the weight limit is reached.

        Returns:
            np.array: Indicates inclusion (1) or exclusion (0) of each item based on their efficiency.
        """
        # Create an array to hold the initial state
        state = np.zeros(self.total_items, dtype=int)
        current_weight = 0

        # Indexes sorted by value to weight ratio (descending)
        item_indexes = np.argsort(-self.item_values / self.item_weights)

        # Add items based on their value-to-weight ratio until the limit is reached
        for idx in item_indexes:
            if current_weight + self.item_weights[idx] <= self.w_limit:
                state[idx] = 1
                current_weight += self.item_weights[idx]
            else:
                break  # Stop adding items if the next item would exceed the weight limit

        return state
    
    def simulated_annealing(self, max_iter, cooling_rate, initial_max_flips, heating_on, time_cutoff, with_greedy_init, restart):
        """
        Implements a simulated annealing algorithm for optimizing a knapsack configuration. 
        It starts with an initial state, which can be either random or based on a greedy algorithm, 
        and iteratively improves the solution by flipping item inclusions based on their impacts on the objective value. 
        The temperature parameter controls the probability of accepting worse solutions, cooling progressively. 
        Optional restarts can reset the state and temperature to escape local minima.

        Parameters:
            max_iter (int): Maximum number of iterations.
            cooling_rate (float): Factor by which temperature is reduced in each iteration.
            initial_max_flips (int): Initial number of maximum flips allowed per iteration.
            heating_on (bool): If True, use a dynamic starting temperature based on state.
            time_cutoff (float): Maximum runtime before termination.
            with_greedy_init (bool): If True, start with a greedy solution.
            restart (bool): If True, allow restarts during optimization.

        Returns:
            tuple: Best state found, its value, its weight, and the total runtime.
        """
        # Get initial time
        start_time = time.time()

        # Set iterations for restart
        if restart:
            restart_iter = 3
        else:
            restart_iter = 1

        if with_greedy_init:
            # Generate inital state with greedy method
            curr_state = self.generate_greedy_feasible_initial_state()
        
        else:
            curr_state = np.zeros(self.total_items, dtype=int)
        
        # Determine starting temperature
        if heating_on:
            T = self.heating_temperature(np.copy(curr_state), initial_max_flips)
        else:
            T = 1000  # Default starting temperature if not using heating phase

        T_start = T
        acceptance_history = []

        if self.verbose:
            print('Initial state: ', curr_state)
            print('Starting temperature: ', T_start)
            print('-----------------------')

        # Initialize best value
        best_state = np.copy(curr_state)
        best_total_value = np.sum(best_state * self.item_values)
        best_total_weight = np.sum(best_state * self.item_weights)

        
        max_flips = initial_max_flips
        i = 0
        old_i = 0

        with open(self.trace_file_name, 'w') as f:
            # Write intial value in trace files
            f.write(f"{self.get_runtime(start_time):.2f},{float(best_total_value)}\n")
            for j in range(restart_iter):
                # Set values if restart is choosen
                if restart:
                    if j == 1:
                        old_i = i
                        i = 0
                        curr_state = np.copy(best_state)
                        max_flips = initial_max_flips
                        T = T_start
                    if j == 2:
                        curr_state = self.generate_greedy_feasible_initial_state()
                        T = T_start
                        max_flips = initial_max_flips
                        i = 0

                while True:

                    # Stopping conditions
                    if T < 0.1 or i > max_iter or self.get_runtime(start_time) > time_cutoff:
                        if self.verbose:  print('Stopped, at {} iterations with a temperature of {} and a runtime of {}'.format((old_i + i), T, self.get_runtime(start_time)))
                        break

                
                    if self.verbose: 
                        print('Current max_flips: ', max_flips)
                    
                    # Get candidate state
                    candidate_state = self.flip_random_element(np.copy(curr_state), max_flips)
                    candidate_total_value = np.sum(candidate_state * self.item_values)
                    candidate_total_weight = np.sum(candidate_state * self.item_weights)

                    if self.verbose:
                        print(f"Iteration {i}: Current state: {curr_state}, Value: {np.sum(curr_state * self.item_values)}, Weight: {np.sum(curr_state * self.item_weights)}")
                        print(f"Candidate state: {candidate_state}, Value: {candidate_total_value}, Weight: {candidate_total_weight}")

                    
                    delta = candidate_total_value - np.sum(curr_state * self.item_values)
                    
                    # Accept better candidates or worse ones based on probability
                    exp_argument = delta / T
                    exp_argument = np.clip(exp_argument, None, 700)  # To avoid overflowing
                    acceptance_prob = np.exp(exp_argument)

                    if (delta > 0 or np.random.rand() < acceptance_prob) and candidate_total_weight <= self.w_limit:
                        if self.verbose: print('Accepting candidate')
                        curr_state = np.copy(candidate_state)
                        if candidate_total_value > best_total_value:
                            best_state = np.copy(candidate_state)
                            best_total_value = np.copy(candidate_total_value)
                            best_total_weight = np.copy(candidate_total_weight)
                            f.write(f"{self.get_runtime(start_time):.2f},{int(best_total_value)}\n")
                    else:
                        if self.verbose: print('Rejecting candidate.')

                    # Update acceptance history
                    accepted = 1 if (delta > 0 or np.random.rand() < acceptance_prob) and candidate_total_weight <= self.w_limit else 0
                    acceptance_history.append(accepted)

                    # Adjust max_flips based on dynamic strategy
                    acceptance_rate = np.mean(acceptance_history[-100:]) if len(acceptance_history) > 100 else 0.5
                    max_flips = self.adjust_max_flips(max_flips, acceptance_rate, min_flips=1)  # Ensure at least 1 flip

                    i += 1
                    # Skip candidate if it exceeds the weight limit
                    if candidate_total_weight > self.w_limit:
                        if self.verbose: print('Candidate exceeds weight limit. Skipping...')
                        continue

                    # Cool down the temperature
                    if self.verbose: print('Current temperature: ', T)
                    T = max(0.1, (T * cooling_rate))



                    if self.verbose: 
                        print('Updated Temperature:', T)
                        print('Current best total value: ', best_total_value)
                        print('--------------------------------')

        # Compute running time
        elapsed_time = self.get_runtime(start_time)
        return best_state, float(best_total_value), float(best_total_weight), elapsed_time

    def generate_solution_file(self, cutoff, randSeed, best_value_found, best_solution_state_found):
        """
        Generates a solution file based on the obtained values

        Parameters:
        cutoff (int): Time limit in seconds for the algorithm.
        randSeed (int or None): Random seed used for the algorithm, if any.
        best_value_found (float): Best objective value found.
        best_solution_state_found (np.array): Best state array indicating item inclusion (1) or exclusion (0).

        Output:
            A text file named according to the format '<file_name>_LS1_<cutoff>_<randSeed>.sol' containing the best value and selected item indices.
        """

        # Obtain the selected_indices list
        selected_indices = np.where(best_solution_state_found == 1)[0] + 1

        # Solution file name
        if randSeed is not None:
            sol_file_name = f"{self.file_name}_LS1_{cutoff}_{randSeed}.sol"
        else:
            sol_file_name = f"{self.file_name}_LS1_{cutoff}.sol"

        # Ensure the directory exists where the files will be saved
        directory = "SA_RESULTS/solution_files"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create and write to the file
        with open(os.path.join(directory, sol_file_name), 'w') as file:
            # Write the quality of the best solution found
            file.write(f"{int(best_value_found)}\n")
            # Write the indices of the numbers selected, comma-separated
            file.write(",".join(map(str, selected_indices)))


    def run(self, max_iter=80000, cooling_rate=0.995, initial_flips=3, heating_on=False, time_cutoff=300, seed=None, with_greedy_init=False, restart=True):
        """
        Runs the simulated annealing algorithm with specified parameters, 
        handles the random seed for reproducibility, and manages output files for solutions and traces. 
        Outputs the best solution found within the given constraints of iterations and time.

        Parameters:
            max_iter (int): Maximum number of iterations for the algorithm.
            cooling_rate (float): Cooling rate for the temperature reduction.
            initial_flips (int): Number of initial flips allowed per iteration.
            heating_on (bool): Indicates if a dynamic initial temperature setting is used.
            time_cutoff (float): Maximum running time allowed for the algorithm.
            seed (int or None): Random seed for generating reproducible results.
            with_greedy_init (bool): Starts the algorithm with a greedy-generated initial state if True.
            restart (bool): Enables restarting the algorithm to potentially escape local minima.

        Returns:
            tuple: The best state found, the best value achieved, and the runtime of the algorithm.
        """
       
        # Sets a seed for testing puposes if passed
        if seed is not None:
            np.random.seed(seed)
            if self.verbose: print('Random seed set to: ', seed)
        
        # Set up trace file directory
        trace_file_dir = "SA_RESULTS/trace_files"
        os.makedirs(trace_file_dir, exist_ok=True)
        self.trace_file_name = f"{trace_file_dir}/{self.file_name}_{'LS1'}_{time_cutoff}_{seed}.trace"
        
        best_state, best_total_value, best_total_weight, elapsed_time = self.simulated_annealing(max_iter, cooling_rate, initial_flips, heating_on, time_cutoff, with_greedy_init, restart)
        
        # Generates solution file
        self.generate_solution_file(time_cutoff, seed, best_total_value, best_state)
        
        if self.verbose:
            print('------------ PARAMETERS USED --------------')
            print('Cooling rate: ', cooling_rate)
            print('Initial flips val: ', initial_flips)
            print('With heating: ', heating_on)
            print('------------ RESULTS --------------')
            print('Best item subset:', best_state)
            print('Best total value:', best_total_value)
            print(f"Simulated annealing runtime: {elapsed_time:.2f} seconds.")
            print(f'weight limit: {self.w_limit} vs solution weight: {best_total_weight}')
        return best_state, best_total_value, elapsed_time

