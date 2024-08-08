import math
import pyomo.environ as pyomo

def household_EMS_initial(delta_T, agent_name, data_time, data_agent):
    """
        Estimates the initial operation of the EMS per household
    """
    
    # Sets
    T = data_time.keys()
    S = data_agent['Scenarios'].keys()
    
    # Variables and parameters
    model = pyomo.ConcreteModel()
    model.x = pyomo.Var(T,S)
    model.x_avg = pyomo.Var(T)
    model.x_std = pyomo.Var(T, domain=pyomo.NonNegativeReals)
    model.power_BESS = pyomo.Var(T, bounds=(-data_agent['Buffer_Power'],data_agent['Buffer_Power']))
    model.energy_BESS = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,data_agent['Buffer_Energy']))
    
    # Initial point
    for t in T:
        model.x_avg[t].value = 0.0
        model.x_std[t].value = 0.0
        for s in S:
            model.x[t,s].value = data_agent['Scenarios'][s]['Timeseries'][t]
            model.x_avg[t].value = model.x_avg[t]() + data_agent['Scenarios'][s]['Probability'] * model.x[t,s]()
        for s in S:
            model.x_std[t].value = model.x_std[t]() + data_agent['Scenarios'][s]['Probability'] * (model.x[t,s]() - model.x_avg[t]())**2
        model.x_std[t].value = math.sqrt(model.x_std[t]())


    # Objective function
    model.of = pyomo.Objective(expr = sum(model.x_avg[t]**2 for t in T), sense=pyomo.minimize)

    # Constraints
    model.cons = pyomo.ConstraintList()
    for t in T:
        model.cons.add(model.x_avg[t] == sum( (data_agent['Scenarios'][s]['Probability'] * model.x[t,s]) for s in S) )
        model.cons.add(model.x_std[t]**2 == sum( (data_agent['Scenarios'][s]['Probability'] * (model.x[t,s] - model.x_avg[t])**2) for s in S ))
        model.cons.add(model.energy_BESS[t] == model.energy_BESS[data_time[t]['ant']] + delta_T * model.power_BESS[t])
        for s in S:
            model.cons.add(model.x[t,s] == model.power_BESS[t] + data_agent['Scenarios'][s]['Timeseries'][t])

    # Optimization process
    solver = pyomo.SolverFactory('ipopt')

    # Results
    try:
        result = solver.solve(model, tee=False)
    except:
        print('Error while executing solver!!!!')
        x_demand_avg = {}
        x_demand_std = {}
        for t in T:
            x_demand_avg[t] = 0
            x_demand_std[t] = 0

        return x_demand_avg, x_demand_std, 100000000

    print('The solution status is ' + result.solver.status + ', and termination condition is ' + result.solver.termination_condition)

    if result.solver.termination_condition == 'optimal':
        fo_solution = round(model.of(),2)
    else:
        fo_solution = 100000000

    print('The local value of the OF is ' + str(fo_solution))

    x_power_BESS = {}
    x_energy_BESS = {}
    x_demand_avg = {}
    x_demand_std = {}
    for t in T:
        x_power_BESS[t] = round(model.power_BESS[t](),2)
        x_energy_BESS[t] = round(model.energy_BESS[t](),2)
        x_demand_avg[t] = round(model.x_avg[t](),2)
        x_demand_std[t] = round(model.x_std[t](),2)

    return x_demand_avg, x_demand_std, fo_solution

def household_EMS_without_limits(delta_T, agent_name, data_time, data_agent, aggregated_demand_avg, demand_agent_avg):
    """
        Optimize the operation of the EMS per household without limits
    """
    
    # Sets
    T = data_time.keys()
    S = data_agent['Scenarios'].keys()
    
    # Variables and parameters
    model = pyomo.ConcreteModel()
    model.x = pyomo.Var(T,S)
    model.x_avg = pyomo.Var(T)
    model.x_std = pyomo.Var(T, domain=pyomo.NonNegativeReals)
    model.power_BESS = pyomo.Var(T, bounds=(-data_agent['Buffer_Power'],data_agent['Buffer_Power']))
    model.energy_BESS = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,data_agent['Buffer_Energy']))
    
    # Initial point
    for t in T:
        model.x_avg[t].value = 0.0
        model.x_std[t].value = 0.0
        for s in S:
            model.x[t,s].value = data_agent['Scenarios'][s]['Timeseries'][t]
            model.x_avg[t].value = model.x_avg[t]() + data_agent['Scenarios'][s]['Probability'] * model.x[t,s]()
        for s in S:
            model.x_std[t].value = model.x_std[t]() + data_agent['Scenarios'][s]['Probability'] * (model.x[t,s]() - model.x_avg[t]())**2
        model.x_std[t].value = math.sqrt(model.x_std[t]())


    # Objective function
    model.of = pyomo.Objective(expr = sum((aggregated_demand_avg[t] - demand_agent_avg[t] + model.x_avg[t])**2 for t in T), sense=pyomo.minimize)

    # Constraints
    model.cons = pyomo.ConstraintList()
    for t in T:
        model.cons.add(model.x_avg[t] == sum( (data_agent['Scenarios'][s]['Probability'] * model.x[t,s]) for s in S) )
        model.cons.add(model.x_std[t]**2 == sum( (data_agent['Scenarios'][s]['Probability'] * (model.x[t,s] - model.x_avg[t])**2) for s in S ))
        model.cons.add(model.energy_BESS[t] == model.energy_BESS[data_time[t]['ant']] + delta_T * model.power_BESS[t])
        for s in S:
            model.cons.add(model.x[t,s] == model.power_BESS[t] + data_agent['Scenarios'][s]['Timeseries'][t])

    # Optimization process
    solver = pyomo.SolverFactory('ipopt')

    # Results
    try:
        result = solver.solve(model, tee=False)
    except:
        print('Error while executing solver!!!!')
        x_demand_avg = {}
        x_demand_std = {}
        for t in T:
            x_demand_avg[t] = 0
            x_demand_std[t] = 0

        return x_demand_avg, x_demand_std, 100000000

    print('The solution status is ' + result.solver.status + ', and termination condition is ' + result.solver.termination_condition)

    if result.solver.termination_condition == 'optimal':
        fo_solution = round(model.of(),2)
    else:
        fo_solution = 100000000

    print('The local value of the OF is ' + str(fo_solution))

    x_power_BESS = {}
    x_energy_BESS = {}
    x_demand_avg = {}
    x_demand_std = {}
    for t in T:
        x_power_BESS[t] = round(model.power_BESS[t](),2)
        x_energy_BESS[t] = round(model.energy_BESS[t](),2)
        x_demand_avg[t] = round(model.x_avg[t](),2)
        x_demand_std[t] = round(model.x_std[t](),2)

    return x_demand_avg, x_demand_std, fo_solution

def household_EMS_with_limits(delta_T, agent_name, data_time, data_agent, aggregated_demand_avg, demand_agent_avg, aggregated_demand_std, demand_agent_std, grid_limit):
    """
        Optimize the operation of the EMS per household with limits
    """
    
    # Sets
    T = data_time.keys()
    S = data_agent['Scenarios'].keys()
    
    # Variables and parameters
    model = pyomo.ConcreteModel()
    model.x = pyomo.Var(T,S)
    model.x_avg = pyomo.Var(T)
    model.x_std = pyomo.Var(T, domain=pyomo.NonNegativeReals)
    model.power_BESS = pyomo.Var(T, bounds=(-data_agent['Buffer_Power'],data_agent['Buffer_Power']))
    model.energy_BESS = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,data_agent['Buffer_Energy']))
    model.theta_1 = pyomo.Var(T, domain=pyomo.NonNegativeReals)
    model.theta_2 = pyomo.Var(T, domain=pyomo.NonNegativeReals)
    
    # Initial point
    for t in T:
        model.x_avg[t].value = 0.0
        model.x_std[t].value = 0.0
        for s in S:
            model.x[t,s].value = data_agent['Scenarios'][s]['Timeseries'][t]
            model.x_avg[t].value = model.x_avg[t]() + data_agent['Scenarios'][s]['Probability'] * model.x[t,s]()
        for s in S:
            model.x_std[t].value = model.x_std[t]() + data_agent['Scenarios'][s]['Probability'] * (model.x[t,s]() - model.x_avg[t]())**2
        model.x_std[t].value = math.sqrt(model.x_std[t]())

    # Objective function
    model.of = pyomo.Objective(expr = sum((model.theta_1[t] + model.theta_2[t]) for t in T), sense=pyomo.minimize)

    # Constraints
    model.cons = pyomo.ConstraintList()
    for t in T:
        model.cons.add(model.theta_1[t] >= (aggregated_demand_avg[t] - demand_agent_avg[t] + model.x_avg[t]) + 2 * (pyomo.sqrt(aggregated_demand_std[t]) - demand_agent_std[t] + model.x_std[t]) - grid_limit)
        model.cons.add(model.theta_2[t] >= -grid_limit - (aggregated_demand_avg[t] - demand_agent_avg[t] + model.x_avg[t]) + 2 * (pyomo.sqrt(aggregated_demand_std[t]) - demand_agent_std[t] + model.x_std[t]))
        # model.cons.add(model.theta_1[t] >= (aggregated_demand_avg[t] - demand_agent_avg[t] + model.x_avg[t]) + 2 * pyomo.sqrt(aggregated_demand_std[t] - demand_agent_std[t]**2 + model.x_std[t]**2) - grid_limit)
        # model.cons.add(model.theta_2[t] >= -grid_limit - (aggregated_demand_avg[t] - demand_agent_avg[t] + model.x_avg[t]) + 2 * pyomo.sqrt(aggregated_demand_std[t] - demand_agent_std[t]**2 + model.x_std[t]**2))
        model.cons.add(model.x_avg[t] == sum( (data_agent['Scenarios'][s]['Probability'] * model.x[t,s]) for s in S) )
        model.cons.add(model.x_std[t]**2 == sum( (data_agent['Scenarios'][s]['Probability'] * (model.x[t,s] - model.x_avg[t])**2) for s in S ))
        model.cons.add(model.energy_BESS[t] == model.energy_BESS[data_time[t]['ant']] + delta_T * model.power_BESS[t])
        for s in S:
            model.cons.add(model.x[t,s] == model.power_BESS[t] + data_agent['Scenarios'][s]['Timeseries'][t])

    # Optimization process
    solver = pyomo.SolverFactory('ipopt')

    # Results
    try:
        result = solver.solve(model, tee=False)
    except:
        print('Error while executing solver!!!!')
        x_demand_avg = {}
        x_demand_std = {}
        for t in T:
            x_demand_avg[t] = 0
            x_demand_std[t] = 0

        return x_demand_avg, x_demand_std, 100000000

    print('The solution status is ' + result.solver.status + ', and termination condition is ' + result.solver.termination_condition)

    if result.solver.termination_condition == 'optimal':
        fo_solution = round(model.of(),2)
    else:
        fo_solution = 100000000

    print('The local value of the OF is ' + str(fo_solution))

    x_power_BESS = {}
    x_energy_BESS = {}
    x_demand_avg = {}
    x_demand_std = {}
    for t in T:
        x_power_BESS[t] = round(model.power_BESS[t](),2)
        x_energy_BESS[t] = round(model.energy_BESS[t](),2)
        x_demand_avg[t] = round(model.x_avg[t](),2)
        x_demand_std[t] = round(model.x_std[t](),2)

    return x_demand_avg, x_demand_std, fo_solution

def household_EMS_total(delta_T, agent_name, data_time, data_agent, aggregated_demand_avg, demand_agent_avg, aggregated_demand_std, demand_agent_std, lower_limit_agent, upper_limit_agent, grid_limit):
    """
        Optimize the operation of the EMS per household with limits
    """
    
    # Sets
    T = data_time.keys()
    S = data_agent['Scenarios'].keys()
    
    # Variables and parameters
    model = pyomo.ConcreteModel()
    model.x = pyomo.Var(T,S)
    model.x_avg = pyomo.Var(T)
    model.x_std = pyomo.Var(T, domain=pyomo.NonNegativeReals)
    model.power_BESS = pyomo.Var(T, bounds=(-data_agent['Buffer_Power'],data_agent['Buffer_Power']))
    model.energy_BESS = pyomo.Var(T, domain=pyomo.NonNegativeReals, bounds=(0,data_agent['Buffer_Energy']))
    
    # Initial point
    for t in T:
        model.x_avg[t].value = 0.0
        model.x_std[t].value = 0.0
        for s in S:
            model.x[t,s].value = data_agent['Scenarios'][s]['Timeseries'][t]
            model.x_avg[t].value = model.x_avg[t]() + data_agent['Scenarios'][s]['Probability'] * model.x[t,s]()
        for s in S:
            model.x_std[t].value = model.x_std[t]() + data_agent['Scenarios'][s]['Probability'] * (model.x[t,s]() - model.x_avg[t]())**2
        model.x_std[t].value = math.sqrt(model.x_std[t]())


    # Objective function
    model.of = pyomo.Objective(expr = sum((aggregated_demand_avg[t] - demand_agent_avg[t] + model.x_avg[t])**2 for t in T), sense=pyomo.minimize)
    
    # Constraints
    model.cons = pyomo.ConstraintList()
    for t in T:
        model.cons.add(model.x_avg[t] + 2 * model.x_std[t] <= upper_limit_agent[t] + 0.01)
        model.cons.add(-lower_limit_agent[t] - 0.01 <= model.x_avg[t] - 2 * model.x_std[t])
        model.cons.add(model.x_avg[t] == sum( (data_agent['Scenarios'][s]['Probability'] * model.x[t,s]) for s in S) )
        model.cons.add(model.x_std[t]**2 == sum( (data_agent['Scenarios'][s]['Probability'] * (model.x[t,s] - model.x_avg[t])**2) for s in S ))
        model.cons.add(model.energy_BESS[t] == model.energy_BESS[data_time[t]['ant']] + delta_T * model.power_BESS[t])
        for s in S:
            model.cons.add(model.x[t,s] == model.power_BESS[t] + data_agent['Scenarios'][s]['Timeseries'][t])

    # Optimization process
    solver = pyomo.SolverFactory('ipopt')

    # Results
    try:
        result = solver.solve(model, tee=False)
    except:
        print('Error while executing solver!!!!')
        x_demand_avg = {}
        x_demand_std = {}
        for t in T:
            x_demand_avg[t] = 0
            x_demand_std[t] = 0

        return x_demand_avg, x_demand_std, 100000000

    print('The solution status is ' + result.solver.status + ', and termination condition is ' + result.solver.termination_condition)

    if result.solver.termination_condition == 'optimal':
        fo_solution = round(model.of(),2)
    else:
        fo_solution = 100000000

    print('The local value of the OF is ' + str(fo_solution))

    x_power_BESS = {}
    x_energy_BESS = {}
    x_demand_avg = {}
    x_demand_std = {}
    for t in T:
        x_power_BESS[t] = round(model.power_BESS[t](),2)
        x_energy_BESS[t] = round(model.energy_BESS[t](),2)
        x_demand_avg[t] = round(model.x_avg[t](),2)
        x_demand_std[t] = round(model.x_std[t](),2)

    return x_demand_avg, x_demand_std, fo_solution