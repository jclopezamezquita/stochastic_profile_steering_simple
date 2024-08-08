import matplotlib.pyplot as plt
import numpy as np

def plot_resulting_profile(time, average_profile, standard_deviation_profile, Max_Y, type_of_plot, fo_value):

    # Converting numerical lists into numpy objects
    average_profile=np.array(average_profile)
    standard_deviation_profile=np.array(standard_deviation_profile)
    
    # Convert time to numerical values for plotting
    time_num = range(len(time))

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_num, average_profile, color='blue', label='Average Profile')

    # Fill the area between average profile +/- standard deviation
    plt.fill_between(time_num, average_profile - 2*standard_deviation_profile, average_profile + 2*standard_deviation_profile, color='lightblue', alpha=0.5, label='Standard Deviation')

    # Add horizontal red line at Max_Y
    plt.axhline(y=Max_Y, color='red', linestyle='--', label=f'Grid Limit = {Max_Y}')
    plt.axhline(y=-Max_Y, color='red', linestyle='--')

    # Set labels and title
    plt.xticks(time_num, time, rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Average Profile with Std: ' + type_of_plot + ', FO: ' + str(fo_value))

    # Calculate plot limits based on Max_Y
    lower_limit = -Max_Y * 1.2
    upper_limit = Max_Y * 1.2

    # Set plot limits
    plt.ylim(lower_limit, upper_limit)

    # Add legend
    plt.legend()

    # Add legend
    plt.grid(True)

    # Save the plot as a PNG image
    name_of_plot = 'average_profile_with_std_' + type_of_plot + '.png'
    plt.savefig('results/' + name_of_plot)

    # plt.show()