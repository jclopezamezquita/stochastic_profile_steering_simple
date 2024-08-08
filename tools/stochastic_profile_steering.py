import math
import json
from tools.household_EMS import *
from tools.plot_time_series import plot_resulting_profile

def collect_data_from_files(time_data_file, agents_data_file, parameters_file):

    # Open JSON file with time data
    with open(time_data_file, 'r') as openfile:
        data_time = json.load(openfile)

    # Open JSON file with agents data
    with open(agents_data_file, 'r') as openfile:
        data_agents = json.load(openfile)

    # Open JSON file with parameters
    with open(parameters_file, 'r') as openfile:
        parameters = json.load(openfile)

    return data_time, data_agents, parameters["DELTA_TIME"], parameters["TOLERANCE"], parameters["LIMIT"] 


def stochastic_profile_steering_initial_profile(time_data_file, agents_data_file, parameters_file):
    """
        Runs Algorithm 0: Initial profile of agents
    """

    print("Running algorithm 1...")

    # Collect input data
    data_time, data_agents, DELTA_TIME, TOLERANCE, LIMIT = collect_data_from_files(time_data_file=time_data_file, agents_data_file=agents_data_file, parameters_file=parameters_file)

    # Show number of households in the EC
    print('There are ' + str(len(data_agents)) + ' households')

    ######################################################### Initial profiles ########################################################################
    
    print("Calculating initial profiles...")
    
    # Initialize result variables
    results = {}
    results['of'] = 0.0
    for agent in data_agents:
        results[agent] = {}
        results[agent]['x_demand_avg'] = {}
        results[agent]['x_demand_std'] = {}
        results[agent]['x_demand_avg_temporal'] = {}
        results[agent]['x_demand_std_temporal'] = {}
        results[agent]['of'] = 0.0
        results[agent]['of_temporal'] = 0.0
        for time in data_time:
            results[agent]['x_demand_avg'][time] = 0.0
            results[agent]['x_demand_std'][time] = 0.0
            results[agent]['x_demand_avg_temporal'][time] = 0.0
            results[agent]['x_demand_std_temporal'][time] = 0.0

    # Run optimization per agent
    for agent in data_agents:
        print("Agent " + agent + ":")
        x_demand_avg, x_demand_std, of = household_EMS_initial(delta_T=DELTA_TIME, agent_name=agent, data_time=data_time, data_agent=data_agents[agent])
        # print(x_demand_avg)
        # print(x_demand_std)

        # Save results
        # results[agent]['of'] = of
        for time in data_time:
            results[agent]['x_demand_avg'][time] = x_demand_avg[time]
            results[agent]['x_demand_std'][time] = x_demand_std[time]

    ######################################################### Average total profile ###################################################################

    results["Aggregated"] = {}
    results["Aggregated"]['x_demand_avg'] = {}
    results["Aggregated"]['x_demand_std'] = {}
    for time in data_time:
        results["Aggregated"]['x_demand_avg'][time] = 0.0
        results["Aggregated"]['x_demand_std'][time] = 0.0
        for agent in data_agents:
            results["Aggregated"]['x_demand_avg'][time] = results["Aggregated"]['x_demand_avg'][time] + results[agent]['x_demand_avg'][time]
            results["Aggregated"]['x_demand_std'][time] = results["Aggregated"]['x_demand_std'][time] + results[agent]['x_demand_std'][time]**2
        results["Aggregated"]['x_demand_std'][time] = math.sqrt(results["Aggregated"]['x_demand_std'][time])
        results["Aggregated"]['x_demand_avg'][time] = round(results["Aggregated"]['x_demand_avg'][time],2)
        results["Aggregated"]['x_demand_std'][time] = round(results["Aggregated"]['x_demand_std'][time],2)
        results['of'] = results['of'] + results["Aggregated"]['x_demand_avg'][time]**2
    results['of'] = round(results['of'],2)
    
    print("Calculating initial average profile...")

    print(results["Aggregated"]['x_demand_avg'])
    print(results["Aggregated"]['x_demand_std'])
    print("Initial OF: " + str(results['of']))

    return results

def stochastic_profile_steering_without_limits(time_data_file, agents_data_file, parameters_file):
    """
        Runs Algorithm 1: SPS without limits
    """

    print("Running algorithm 1...")

    # Collect input data
    data_time, data_agents, DELTA_TIME, TOLERANCE, LIMIT = collect_data_from_files(time_data_file=time_data_file, agents_data_file=agents_data_file, parameters_file=parameters_file)

    # Show number of households in the EC
    print('There are ' + str(len(data_agents)) + ' households')

    ######################################################### Line 1 - Algorithm 1 ####################################################################
    ######################################################### Initial profiles ########################################################################
    
    print("Calculating initial profiles...")
    
    # Initialize result variables
    results = {}
    results['of'] = 0.0
    for agent in data_agents:
        results[agent] = {}
        results[agent]['x_demand_avg'] = {}
        results[agent]['x_demand_std'] = {}
        results[agent]['x_demand_avg_temporal'] = {}
        results[agent]['x_demand_std_temporal'] = {}
        results[agent]['of'] = 0.0
        results[agent]['of_temporal'] = 0.0
        for time in data_time:
            results[agent]['x_demand_avg'][time] = 0.0
            results[agent]['x_demand_std'][time] = 0.0
            results[agent]['x_demand_avg_temporal'][time] = 0.0
            results[agent]['x_demand_std_temporal'][time] = 0.0

    # Run optimization per agent
    for agent in data_agents:
        print("Agent " + agent + ":")
        x_demand_avg, x_demand_std, of = household_EMS_initial(delta_T=DELTA_TIME, agent_name=agent, data_time=data_time, data_agent=data_agents[agent])
        # print(x_demand_avg)
        # print(x_demand_std)

        # Save results
        # results[agent]['of'] = of
        for time in data_time:
            results[agent]['x_demand_avg'][time] = x_demand_avg[time]
            results[agent]['x_demand_std'][time] = x_demand_std[time]

    ######################################################### Line 2 - Algorithm 1 ####################################################################
    ######################################################### Average total profile ###################################################################

    results["Aggregated"] = {}
    results["Aggregated"]['x_demand_avg'] = {}
    results["Aggregated"]['x_demand_std'] = {}
    for time in data_time:
        results["Aggregated"]['x_demand_avg'][time] = 0.0
        results["Aggregated"]['x_demand_std'][time] = 0.0
        for agent in data_agents:
            results["Aggregated"]['x_demand_avg'][time] = results["Aggregated"]['x_demand_avg'][time] + results[agent]['x_demand_avg'][time]
            results["Aggregated"]['x_demand_std'][time] = results["Aggregated"]['x_demand_std'][time] + results[agent]['x_demand_std'][time]**2
        # results["Aggregated"]['x_demand_std'][time] = math.sqrt(results["Aggregated"]['x_demand_std'][time])
        results["Aggregated"]['x_demand_avg'][time] = round(results["Aggregated"]['x_demand_avg'][time],2)
        results["Aggregated"]['x_demand_std'][time] = round(results["Aggregated"]['x_demand_std'][time],2)
        results['of'] = results['of'] + results["Aggregated"]['x_demand_avg'][time]**2
    results['of'] = round(results['of'],2)
    
    print("Calculating initial average profile...")
    print("Average profile: " + str([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time]))
    print("Std. deviation: " + str([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time]))
    print("Initial OF - Algorithm 1: " + str(results['of']))

    # Plot results
    plot_resulting_profile(time=data_time, 
        average_profile=[round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time], 
        standard_deviation_profile=[round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time],
        Max_Y=LIMIT, type_of_plot='initial', fo_value=results['of'])

    ######################################################### Line 3 - Algorithm 1 #####################################################################
    ######################################################### Initialize improvement ###################################################################

    total_improvement = 1000000

    ######################################################### Line 4 - Algorithm 1 ####################################################################
    ######################################################### while loop ##############################################################################

    iteration = 0
    while total_improvement > TOLERANCE:
        iteration = iteration + 1

        ######################################################### Line 5 - Algorithm 1 ####################################################################
        ######################################################### for loop ################################################################################

        improvement_per_agent = {}
        for agent in data_agents:
            improvement_per_agent[agent] = 0.0

            ######################################################### Line 6 - Algorithm 1 ####################################################################
            ######################################################### Optimize average consumption per agent ##################################################

            print("Agent " + agent + ":")
            x_demand_avg, x_demand_std, of = household_EMS_without_limits(delta_T=DELTA_TIME, agent_name=agent, data_time=data_time, data_agent=data_agents[agent], aggregated_demand_avg=results["Aggregated"]['x_demand_avg'], demand_agent_avg=results[agent]['x_demand_avg'])

            # Save results
            results[agent]['of_temporal'] = of
            for time in data_time:
                results[agent]['x_demand_avg_temporal'][time] = x_demand_avg[time]
                results[agent]['x_demand_std_temporal'][time] = x_demand_std[time]

            ######################################################### Line 7 - Algorithm 1 ####################################################################
            ######################################################### Calculate improvement per agent  ########################################################

            improvement_per_agent[agent] = round(results['of'] - results[agent]['of_temporal'],2)
        
        ######################################################### Line 9 - Algorithm 1 ####################################################################
        ######################################################### Determine the best agent based on improvements  #########################################

        best_improvement_per_agent = 0.0
        best_agent = None
        for agent in data_agents:
            if best_improvement_per_agent < improvement_per_agent[agent]:
                best_improvement_per_agent = improvement_per_agent[agent]
                best_agent = agent
        if best_agent is None:
            print("Best agent - Algorithm 3: None")
            break
        else:
            print("Best agent - Algorithm 3: " + best_agent)
            total_improvement = best_improvement_per_agent
            print("Total improvement: " + str(total_improvement))

        ######################################################### Line 10 - Algorithm 1 ####################################################################
        ######################################################### Update the profile of the best agent  ####################################################

        # Update best agent profile anf OF
        for time in data_time:
            results[best_agent]['x_demand_avg'][time] = results[best_agent]['x_demand_avg_temporal'][time]
            results[best_agent]['x_demand_std'][time] = results[best_agent]['x_demand_std_temporal'][time]

        # Update aggregated profile
        results['of'] = 0.0
        for time in data_time:
            results["Aggregated"]['x_demand_avg'][time] = 0.0
            results["Aggregated"]['x_demand_std'][time] = 0.0
            for agent in data_agents:
                results["Aggregated"]['x_demand_avg'][time] = results["Aggregated"]['x_demand_avg'][time] + results[agent]['x_demand_avg'][time]
                results["Aggregated"]['x_demand_std'][time] = results["Aggregated"]['x_demand_std'][time] + results[agent]['x_demand_std'][time]**2
            # results["Aggregated"]['x_demand_std'][time] = math.sqrt(results["Aggregated"]['x_demand_std'][time])
            results["Aggregated"]['x_demand_avg'][time] = round(results["Aggregated"]['x_demand_avg'][time],2)
            results["Aggregated"]['x_demand_std'][time] = round(results["Aggregated"]['x_demand_std'][time],2)
            results['of'] = results['of'] + results["Aggregated"]['x_demand_avg'][time]**2
        results['of'] = round(results['of'],2)
        
        print("Calculating updated average profile...")

        print("Average profile: " + str([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time]))
        print("Std. deviation: " + str([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time]))
        print("Iteration " + str(iteration) + ", updated OF - Algorithm 1: " + str(results['of']))


    ######################################################### Line 12 - Algorithm 1 ####################################################################
    ######################################################### Show the optimized profile  ####################################################

    print("Calculating final average profile...")

    print([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time])
    print([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time])
    print("Final OF - Algorithm 1: " + str(results['of']))

    # Plot results
    plot_resulting_profile(time=data_time, 
        average_profile=[round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time], 
        standard_deviation_profile=[round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time],
        Max_Y=LIMIT, type_of_plot='SPS_without_limits', fo_value=results['of'])

    return results

def stochastic_profile_steering_with_limits(time_data_file, agents_data_file, parameters_file):
    """
        Runs Algorithm 2: SPS with limits, disregarding the community objective
    """

    # Collect input data
    data_time, data_agents, DELTA_TIME, TOLERANCE, LIMIT = collect_data_from_files(time_data_file=time_data_file, agents_data_file=agents_data_file, parameters_file=parameters_file)

    # Show number of households in the EC
    print('There are ' + str(len(data_agents)) + ' households')

    ######################################################### Line 1 - Algorithm 2 ####################################################################
    ######################################################### Initial profiles ########################################################################
    
    print("Calculating initial profiles...")
    
    # Initialize result variables
    results = {}
    results['of'] = 0.0
    for agent in data_agents:
        results[agent] = {}
        results[agent]['x_demand_avg'] = {}
        results[agent]['x_demand_std'] = {}
        results[agent]['x_demand_avg_temporal'] = {}
        results[agent]['x_demand_std_temporal'] = {}
        results[agent]['of'] = 0.0
        results[agent]['of_temporal'] = 0.0
        for time in data_time:
            results[agent]['x_demand_avg'][time] = 0.0
            results[agent]['x_demand_std'][time] = 0.0
            results[agent]['x_demand_avg_temporal'][time] = 0.0
            results[agent]['x_demand_std_temporal'][time] = 0.0

    # Run optimization per agent
    for agent in data_agents:
        print("Agent " + agent + ":")
        x_demand_avg, x_demand_std, of = household_EMS_initial(delta_T=DELTA_TIME, agent_name=agent, data_time=data_time, data_agent=data_agents[agent])
        # print(x_demand_avg)
        # print(x_demand_std)

        # Save results
        # results[agent]['of'] = of
        for time in data_time:
            results[agent]['x_demand_avg'][time] = x_demand_avg[time]
            results[agent]['x_demand_std'][time] = x_demand_std[time]

    ######################################################### Line 2 - Algorithm 2 ####################################################################
    ######################################################### Average profile and Standard Deviation ##################################################

    results["Aggregated"] = {}
    results["Aggregated"]['x_demand_avg'] = {}
    results["Aggregated"]['x_demand_std'] = {}
    for time in data_time:
        results["Aggregated"]['x_demand_avg'][time] = 0.0
        results["Aggregated"]['x_demand_std'][time] = 0.0
        for agent in data_agents:
            results["Aggregated"]['x_demand_avg'][time] = results["Aggregated"]['x_demand_avg'][time] + results[agent]['x_demand_avg'][time]
            results["Aggregated"]['x_demand_std'][time] = results["Aggregated"]['x_demand_std'][time] + results[agent]['x_demand_std'][time]**2
        # results["Aggregated"]['x_demand_std'][time] = math.sqrt(results["Aggregated"]['x_demand_std'][time])
        results["Aggregated"]['x_demand_avg'][time] = round(results["Aggregated"]['x_demand_avg'][time],2)
        results["Aggregated"]['x_demand_std'][time] = round(results["Aggregated"]['x_demand_std'][time],2)
        results['of'] = results['of'] + max(results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(results["Aggregated"]['x_demand_std'][time]) - LIMIT, -LIMIT - results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(results["Aggregated"]['x_demand_std'][time]), 0)
    results['of'] = round(results['of'],2)
    
    print("Calculating initial average profile...")

    print("Average profile: " + str([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time]))
    print("Std. deviation: " + str([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time]))
    print("Initial OF - Algorithm 2: " + str(results['of']))

    ######################################################### Line 3 - Algorithm 2 #####################################################################
    ######################################################### Initialize improvement ###################################################################

    total_improvement = 1000000

    ######################################################### Line 4 - Algorithm 2 ####################################################################
    ######################################################### while loop ##############################################################################

    iteration = 0
    while total_improvement > TOLERANCE:
        iteration = iteration + 1

        ######################################################### Line 5 - Algorithm 2 ####################################################################
        ######################################################### for loop ################################################################################

        improvement_per_agent = {}
        for agent in data_agents:
            improvement_per_agent[agent] = 0.0

            ######################################################### Line 6 - Algorithm 2 ####################################################################
            ######################################################### Optimize average consumption per agent ##################################################

            print("Agent " + agent + ":")
            x_demand_avg, x_demand_std, of = household_EMS_with_limits(delta_T=DELTA_TIME, agent_name=agent, data_time=data_time, data_agent=data_agents[agent], aggregated_demand_avg=results["Aggregated"]['x_demand_avg'], demand_agent_avg=results[agent]['x_demand_avg'], aggregated_demand_std=results["Aggregated"]['x_demand_std'], demand_agent_std=results[agent]['x_demand_std'], grid_limit=LIMIT)

            # Save results
            results[agent]['of_temporal'] = of
            for time in data_time:
                results[agent]['x_demand_avg_temporal'][time] = x_demand_avg[time]
                results[agent]['x_demand_std_temporal'][time] = x_demand_std[time]

            ######################################################### Line 7 - Algorithm 2 ####################################################################
            ######################################################### Calculate improvement per agent  ########################################################

            improvement_per_agent[agent] = round(results['of'] - results[agent]['of_temporal'],2)
        
        ######################################################### Line 9 - Algorithm 2 ####################################################################
        ######################################################### Determine the best agent based on improvements  #########################################

        best_improvement_per_agent = 0.0
        best_agent = None
        for agent in data_agents:
            if best_improvement_per_agent < improvement_per_agent[agent]:
                best_improvement_per_agent = improvement_per_agent[agent]
                best_agent = agent
        if best_agent is None:
            print("Best agent - Algorithm 3: None")
            break
        else:
            print("Best agent - Algorithm 3: " + best_agent)
            total_improvement = best_improvement_per_agent
            print("Total improvement: " + str(total_improvement))

        ######################################################### Line 10 - Algorithm 2 ####################################################################
        ######################################################### Update the profile of the best agent  ####################################################

        # Update best agent profile anf OF
        for time in data_time:
            results[best_agent]['x_demand_avg'][time] = results[best_agent]['x_demand_avg_temporal'][time]
            results[best_agent]['x_demand_std'][time] = results[best_agent]['x_demand_std_temporal'][time]

        # Update aggregated profile
        results['of'] = 0.0
        for time in data_time:
            results["Aggregated"]['x_demand_avg'][time] = 0.0
            results["Aggregated"]['x_demand_std'][time] = 0.0
            for agent in data_agents:
                results["Aggregated"]['x_demand_avg'][time] = results["Aggregated"]['x_demand_avg'][time] + results[agent]['x_demand_avg'][time]
                results["Aggregated"]['x_demand_std'][time] = results["Aggregated"]['x_demand_std'][time] + results[agent]['x_demand_std'][time]**2
            # results["Aggregated"]['x_demand_std'][time] = math.sqrt(results["Aggregated"]['x_demand_std'][time])
            results["Aggregated"]['x_demand_avg'][time] = round(results["Aggregated"]['x_demand_avg'][time],2)
            results["Aggregated"]['x_demand_std'][time] = round(results["Aggregated"]['x_demand_std'][time],2)
            results['of'] = results['of'] + max(results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(results["Aggregated"]['x_demand_std'][time]) - LIMIT, -LIMIT - results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(results["Aggregated"]['x_demand_std'][time]), 0)
        results['of'] = round(results['of'],2)

        print("Calculating updated average profile...")

        print("Average profile: " + str([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time]))
        print("Std. deviation: " + str([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time]))
        print("Iteration " + str(iteration) + ", updated OF - Algorithm 2: " + str(results['of']))
        
    ######################################################### Line 12 - Algorithm 2 ####################################################################
    ######################################################### Show the optimized profile  ####################################################

    print("Calculating final average profile...")

    print("Average profile: " + str([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time]))
    print("Std. deviation: " + str([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time]))
    print("Final OF - Algorithm 2: " + str(results['of']))

    # Plot results
    plot_resulting_profile(time=data_time, 
        average_profile=[round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time], 
        standard_deviation_profile=[round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time],
        Max_Y=LIMIT, type_of_plot='SPS_only_limits', fo_value=results['of'])


    return results

def stochastic_profile_steering(time_data_file, agents_data_file, parameters_file):
    """
        Runs Algorithm 3: SPS with limits and considering community objective
    """

    # Collect input data
    data_time, data_agents, DELTA_TIME, TOLERANCE, LIMIT = collect_data_from_files(time_data_file=time_data_file, agents_data_file=agents_data_file, parameters_file=parameters_file)

    # Show number of households in the EC
    print('There are ' + str(len(data_agents)) + ' households in the EC')

    ######################################################### Line 1 - Algorithm 3 ####################################################################
    ######################################################### Initial profiles and Algortihm 1 ########################################################################
    
    print("Calculating initial profiles...")
    
    # Initialize result variables
    results = {}
    results['of'] = 0.0
    for agent in data_agents:
        results[agent] = {}
        results[agent]['x_demand_avg'] = {}
        results[agent]['x_demand_std'] = {}
        results[agent]['x_demand_avg_temporal'] = {}
        results[agent]['x_demand_std_temporal'] = {}
        results[agent]['lower_individual_limit'] = {}
        results[agent]['upper_individual_limit'] = {}
        results[agent]['of'] = 0.0
        results[agent]['of_temporal'] = 0.0
        for time in data_time:
            results[agent]['x_demand_avg'][time] = 0.0
            results[agent]['x_demand_std'][time] = 0.0
            results[agent]['x_demand_avg_temporal'][time] = 0.0
            results[agent]['x_demand_std_temporal'][time] = 0.0
            results[agent]['lower_individual_limit'][time] = 0.0
            results[agent]['upper_individual_limit'][time] = 0.0

    # Call algorithm 1
    print("############## Starting Algorithm 1 ######################")
    initial_results = stochastic_profile_steering_without_limits(time_data_file, agents_data_file, parameters_file)
    print("############## Algorithm 1 has finished ######################")    
    for agent in data_agents:
        for time in data_time:
            results[agent]['x_demand_avg'][time] = initial_results[agent]['x_demand_avg'][time]
            results[agent]['x_demand_std'][time] = initial_results[agent]['x_demand_std'][time]


    ######################################################### Average Profile and Standard Deviation ##################################################

    results["Aggregated"] = {}
    results["Aggregated"]['x_demand_avg'] = {}
    results["Aggregated"]['x_demand_std'] = {}
    for time in data_time:
        results["Aggregated"]['x_demand_avg'][time] = 0.0
        results["Aggregated"]['x_demand_std'][time] = 0.0
        for agent in data_agents:
            results["Aggregated"]['x_demand_avg'][time] = results["Aggregated"]['x_demand_avg'][time] + results[agent]['x_demand_avg'][time]
            results["Aggregated"]['x_demand_std'][time] = results["Aggregated"]['x_demand_std'][time] + results[agent]['x_demand_std'][time]**2
        # results["Aggregated"]['x_demand_std'][time] = math.sqrt(results["Aggregated"]['x_demand_std'][time])
        results["Aggregated"]['x_demand_avg'][time] = round(results["Aggregated"]['x_demand_avg'][time],2)
        results["Aggregated"]['x_demand_std'][time] = round(results["Aggregated"]['x_demand_std'][time],2)
        results['of'] = results['of'] + max(results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(results["Aggregated"]['x_demand_std'][time]) - LIMIT, -LIMIT - results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(results["Aggregated"]['x_demand_std'][time]), 0)
    results['of'] = round(results['of'],2)
    
    print("Calculating probablistic grid limit violations...")

    print("Average profile: " + str([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time]))
    print("Std. deviation: " + str([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time]))
    print("Total grid limit violations: " + str(results['of']))

    ######################################################### Lines 2 - 3 - Algorithm 3 #####################################################################
    ######################################################### Check probabilistic grid limits ###############################################################

    # Verifies if there are limit violations
    if results['of'] > 0.1:

        # Call algorithm 2
        print("############## Starting Algorithm 2 ######################")
        initial_results = stochastic_profile_steering_with_limits(time_data_file, agents_data_file, parameters_file)
        print("############## Algorithm 2 has finished ######################")

        # Estimate individual grid limits
        for agent in data_agents:
            for time in data_time:
                LIMIT_UP = 0.0
                LIMIT_DOWN = 0.0
                if initial_results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(initial_results["Aggregated"]['x_demand_std'][time]) > LIMIT:
                    LIMIT_UP = initial_results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(initial_results["Aggregated"]['x_demand_std'][time])
                else:
                    LIMIT_UP = LIMIT
                if -LIMIT > initial_results["Aggregated"]['x_demand_avg'][time] - 2 * math.sqrt(initial_results["Aggregated"]['x_demand_std'][time]):
                    LIMIT_DOWN = -initial_results["Aggregated"]['x_demand_avg'][time] + 2 * math.sqrt(initial_results["Aggregated"]['x_demand_std'][time])
                else:
                    LIMIT_DOWN = LIMIT

                # results[agent]['upper_individual_limit'][time] = round(LIMIT_UP - (initial_results["Aggregated"]['x_demand_avg'][time] - initial_results[agent]['x_demand_avg'][time]) - 2 * math.sqrt(max(0,initial_results["Aggregated"]['x_demand_std'][time] - round(initial_results[agent]['x_demand_std'][time]**2,2))),2)
                # results[agent]['lower_individual_limit'][time] = round(LIMIT_DOWN + (initial_results["Aggregated"]['x_demand_avg'][time] - initial_results[agent]['x_demand_avg'][time]) - 2 * math.sqrt(max(0,initial_results["Aggregated"]['x_demand_std'][time] - round(initial_results[agent]['x_demand_std'][time]**2,2))),2)
                results[agent]['upper_individual_limit'][time] = round(LIMIT_UP - (initial_results["Aggregated"]['x_demand_avg'][time] - initial_results[agent]['x_demand_avg'][time]) - 2 * max(0,math.sqrt(initial_results["Aggregated"]['x_demand_std'][time]) - round(initial_results[agent]['x_demand_std'][time],2)),2)
                results[agent]['lower_individual_limit'][time] = round(LIMIT_DOWN + (initial_results["Aggregated"]['x_demand_avg'][time] - initial_results[agent]['x_demand_avg'][time]) - 2 * max(0,math.sqrt(initial_results["Aggregated"]['x_demand_std'][time]) - round(initial_results[agent]['x_demand_std'][time],2)),2)
            # print(results[agent]['upper_individual_limit'])
            # print(results[agent]['lower_individual_limit'])
        
        print("############## Starting Algorithm 3 - SPS ######################")

        # Update state variables and of
        results['of'] = 0.0
        for time in data_time:
            results["Aggregated"]['x_demand_avg'][time] = 0.0
            results["Aggregated"]['x_demand_std'][time] = 0.0
            for agent in data_agents:
                results[agent]['x_demand_avg'][time] = initial_results[agent]['x_demand_avg'][time]
                results[agent]['x_demand_std'][time] = initial_results[agent]['x_demand_std'][time]
                results["Aggregated"]['x_demand_avg'][time] = results["Aggregated"]['x_demand_avg'][time] + results[agent]['x_demand_avg'][time]
                results["Aggregated"]['x_demand_std'][time] = results["Aggregated"]['x_demand_std'][time] + results[agent]['x_demand_std'][time]**2
            # results["Aggregated"]['x_demand_std'][time] = math.sqrt(results["Aggregated"]['x_demand_std'][time])
            results["Aggregated"]['x_demand_avg'][time] = round(results["Aggregated"]['x_demand_avg'][time],2)
            results["Aggregated"]['x_demand_std'][time] = round(results["Aggregated"]['x_demand_std'][time],2)
            results['of'] = results['of'] + results["Aggregated"]['x_demand_avg'][time]**2
        results['of'] = round(results['of'],2)
        
        ######################################################### Line 8 - Algorithm 3 #####################################################################
        ######################################################### Initialize improvement ###################################################################

        total_improvement = 1000000

        ######################################################### Line 9 - Algorithm 3 ####################################################################
        ######################################################### while loop ##############################################################################

        iteration = 0
        while total_improvement > TOLERANCE:
            iteration = iteration + 1

            ######################################################### Line 10 - Algorithm 3 ####################################################################
            ######################################################### for loop ################################################################################

            improvement_per_agent = {}
            for agent in data_agents:
                improvement_per_agent[agent] = 0.0

                ######################################################### Lines 11-12 - Algorithm 3 ####################################################################
                ######################################################### Optimize average consumption per agent ##################################################

                print("Agent " + agent + ":")
                x_demand_avg, x_demand_std, of = household_EMS_total(delta_T=DELTA_TIME, agent_name=agent, data_time=data_time, data_agent=data_agents[agent], aggregated_demand_avg=results["Aggregated"]['x_demand_avg'], demand_agent_avg=results[agent]['x_demand_avg'], aggregated_demand_std=results["Aggregated"]['x_demand_std'], demand_agent_std=results[agent]['x_demand_std'], lower_limit_agent=results[agent]['lower_individual_limit'], upper_limit_agent=results[agent]['upper_individual_limit'], grid_limit=LIMIT)

                # Save results
                results[agent]['of_temporal'] = of
                for time in data_time:
                    results[agent]['x_demand_avg_temporal'][time] = x_demand_avg[time]
                    results[agent]['x_demand_std_temporal'][time] = x_demand_std[time]

                ######################################################### Line 13 - Algorithm 3 ####################################################################
                ######################################################### Calculate improvement per agent  ########################################################

                improvement_per_agent[agent] = round(results['of'] - results[agent]['of_temporal'],2)
            
            ######################################################### Line 15 - Algorithm 3 ####################################################################
            ######################################################### Determine the best agent based on improvements  #########################################

            best_improvement_per_agent = 0.0
            best_agent = None
            for agent in data_agents:
                if best_improvement_per_agent < improvement_per_agent[agent]:
                    best_improvement_per_agent = improvement_per_agent[agent]
                    best_agent = agent
            if best_agent is None:
                print("Best agent - Algorithm 3: None")
                break
            else:
                print("Best agent - Algorithm 3: " + best_agent)
                total_improvement = best_improvement_per_agent
                print("Total improvement: " + str(total_improvement))

            ######################################################### Line 16 - Algorithm 3 ####################################################################
            ######################################################### Update the profile of the best agent  ####################################################

            # Update best agent profile anf OF
            for time in data_time:
                results[best_agent]['x_demand_avg'][time] = results[best_agent]['x_demand_avg_temporal'][time]
                results[best_agent]['x_demand_std'][time] = results[best_agent]['x_demand_std_temporal'][time]

            # Update aggregated profile
            results['of'] = 0.0
            for time in data_time:
                results["Aggregated"]['x_demand_avg'][time] = 0.0
                results["Aggregated"]['x_demand_std'][time] = 0.0
                for agent in data_agents:
                    results["Aggregated"]['x_demand_avg'][time] = results["Aggregated"]['x_demand_avg'][time] + results[agent]['x_demand_avg'][time]
                    results["Aggregated"]['x_demand_std'][time] = results["Aggregated"]['x_demand_std'][time] + results[agent]['x_demand_std'][time]**2
                # results["Aggregated"]['x_demand_std'][time] = math.sqrt(results["Aggregated"]['x_demand_std'][time])
                results["Aggregated"]['x_demand_avg'][time] = round(results["Aggregated"]['x_demand_avg'][time],2)
                results["Aggregated"]['x_demand_std'][time] = round(results["Aggregated"]['x_demand_std'][time],2)
                results['of'] = results['of'] + results["Aggregated"]['x_demand_avg'][time]**2
            results['of'] = round(results['of'],2)

            print("Calculating updated average profile...")

            print("Average profile: " + str([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time]))
            print("Std. deviation: " + str([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time]))
            print("Iteration " + str(iteration) + ", updated OF - Algorithm 3: " + str(results['of']))
            
    ######################################################### Line 18 - Algorithm 3 ####################################################################
    ######################################################### Show the optimized profile  ####################################################

    print("Calculating final average profile...")

    print("Average profile: " + str([round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time]))
    print("Std. deviation: " + str([round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time]))
    print("Final OF - Algorithm 3: " + str(results['of']))

    # Plot results
    plot_resulting_profile(time=data_time, 
        average_profile=[round(results["Aggregated"]['x_demand_avg'][time],2) for time in data_time], 
        standard_deviation_profile=[round(math.sqrt(results["Aggregated"]['x_demand_std'][time]),2) for time in data_time],
        Max_Y=LIMIT, type_of_plot='SPS_final', fo_value=results['of'])

    return results