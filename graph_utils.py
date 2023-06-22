from rdflib import Graph
import numpy as np
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

def calc_rdf_info(dataset):
    """This function calculates information about the RDF dataset"""

    # Loading the RDF dataset
    g = Graph()
    g.parse(dataset, format="nt")

    # Counting the number of assertions
    num_assertions = len(g)

    # Counting the number of unique relations
    unique_relations = set(p for s, p, o in g)
    num_relations = len(unique_relations)

    # Counting the number of unique classes
    unique_classes = set(o for s, p, o in g if p == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    num_classes = len(unique_classes)

    # Counting the number of unique entities
    unique_entities = set(s for s, _, _ in g) | set(o for _, _, o in g)
    num_entities = len(unique_entities)

    # Counting the number of unique literals
    unique_literals = set(o for _, _, o in g if isinstance(o, (str, int, float)))
    num_literals = len(unique_literals)

    print("Number of assertions: {}".format(num_assertions))
    print("Number of relations: {}".format(num_relations))
    print("Number of labeled classes: {}".format(num_classes))
    print("Number of entities: {}".format(num_entities))
    print("Number of literals: {}".format(num_literals))


def calc_statistics(data):
    """This function calculates the average and standard deviation of the given data"""

    # Calculating the average value
    average = round(np.mean(data), 4)

    # Calculating the standard deviation
    std_dev = round(np.std(data), 4)

    print(average, "±", std_dev)

    
def anova_test():
    """This function performs an ANOVA test on the combined data from different datasets"""

    dataset1 = np.array([
        [0.71, 0.6919, 0.6836, 0.6862, 0.6807, 0.6814, 0.7124, 0.697, 0.688, 0.6878],
        [0.7516, 0.5018, 0.4673, 0.4726, 0.4634, 0.4644, 0.7694, 0.696, 0.4778, 0.4744],
        [0.72, 0.569, 0.5551, 0.5592, 0.5514, 0.5523, 0.7236, 0.6817, 0.5629, 0.5612],
        [0.8565, 0.7459, 0.4782, 0.5396, 0.4187, 0.4324, 0.8663, 0.8707, 0.8613, 0.6395]
    ])

    dataset2 = np.array([
        [0.8987, 0.886, 0.5979, 0.8288, 0.7591, 0.7048, 0.8969, 0.8969, 0.8521, 0.8336],
        [0.9136, 0.907, 0.4815, 0.7902, 0.7446, 0.6323, 0.9123, 0.9122, 0.888, 0.7968],
        [0.8958, 0.8837, 0.496, 0.7956, 0.7248, 0.6392, 0.894, 0.8938, 0.8514, 0.8002],
        [0.6238, 0.6892, 0.4802, 0.5662, 0.6168, 0.5395, 0.6304, 0.6217, 0.6226, 0.5962]
    ])

    dataset3 = np.array([
        [0.5504, 0.4856, 0.5106, 0.5706, 0.5292, 0.5296, 0.5281, 0.49, 0.5385, 0.5305],
        [0.5687, 0.3094, 0.2769, 0.6751, 0.28, 0.2805, 0.5532, 0.3303, 0.3803, 0.6036],
        [0.4069, 0.3362, 0.3579, 0.4743, 0.3663, 0.3667, 0.4105, 0.3383, 0.3918, 0.4238],
        [0.41, 0.3128, 0.4218, 0.4146, 0.442, 0.4418, 0.3486, 0.3472, 0.417, 0.3508]
    ])

    # Combining the data from different datasets
    combined_data = np.concatenate((dataset1, dataset2, dataset3), axis=1)

    # Performing the ANOVA test
    f_value, p_value = f_oneway(*combined_data)

    # Print the results
    print("ANOVA Test Results:")
    print("F-value:", f_value)
    print("p-value:", p_value)


def performance_plotting():
    """
    This function creates a line plot of the performance metrics for different sampling strategies
    (The data for the other datasets can be found in results.txt)
    """

    # The average performance metrics of AIFB+ per sampling strategy
    accuracy = [0.71, 0.6919, 0.6836, 0.6862, 0.6807, 0.6814, 0.7124, 0.697, 0.688, 0.6878]
    precision = [0.7516, 0.5018, 0.4673, 0.4726, 0.4634, 0.4644, 0.7694, 0.696, 0.4778, 0.4744]
    f1_score = [0.72, 0.569, 0.5551, 0.5592, 0.5514, 0.5523, 0.7236, 0.6817, 0.5629, 0.5612]
    roc_auc = [0.8565, 0.7459, 0.4782, 0.5396, 0.4187, 0.4324, 0.8663, 0.8707, 0.8613, 0.6395]

    # The standard deviations for the average performance metrics of AIFB+ per sampling strategy
    accuracy_std = [0.0035, 0.0065, 0.0081, 0.0089, 0.0071, 0.0058, 0.0262, 0.0127, 0.011, 0.01]
    precision_std = [0.0063, 0.0311, 0.011, 0.0131, 0.0098, 0.0079, 0.0457, 0.058, 0.0166, 0.0143]
    f1_score_std = [0.0036, 0.0099, 0.0105, 0.0119, 0.0092, 0.0075, 0.0251, 0.0443, 0.0149, 0.0132]
    roc_auc_std = [0.0192, 0.2304, 0.0341, 0.2655, 0.1486, 0.0651, 0.0171, 0.0128, 0.0082, 0.195]

    sampling_strategies = ['RNS', 'NTS', 'ETS', 'NTETS', 'DBS', 'DCS', 'PRS', 'NTPRS', 'ETPRS', 'NTDBS']

    fig, ax = plt.subplots()

    # Plotting the performance metrics with error bars
    ax.errorbar(sampling_strategies, accuracy, yerr=accuracy_std, label='Accuracy', marker='o', alpha=0.2)
    ax.errorbar(sampling_strategies, precision, yerr=precision_std, label='Precision', marker='o', alpha=0.2)
    ax.errorbar(sampling_strategies, f1_score, yerr=f1_score_std, label='F1 Score', marker='o', alpha=0.2)
    ax.errorbar(sampling_strategies, roc_auc, yerr=roc_auc_std, label='ROC AUC', marker='o', alpha=0.2)

    # Setting axis labels and title
    ax.set_xlabel('Sampling Strategies')
    ax.set_ylabel('Performance Metrics')
    ax.set_title('Performance Metrics for Different Sampling Strategies (AIFB+)')

    ax.legend()
    plt.xticks(rotation=45)

    for errorbar in ax.lines[-4:]:
        errorbar.set_alpha(1)

    plt.show()
    

def execution_time_plotting():
    """This function creates a line plot of the execution time for different sampling strategies per dataset"""
    
    # The average execution times for each dataset per sampling strategy
    aifb = [9.192, 6.326, 5.082, 6.206, 6.164, 5.138, 9.43, 8.708, 8.536, 6.736]
    anime = [114.626, 121.908, 66.55, 91.65, 152.258, 86.854, 114.542, 130.246, 111.512, 154.51]
    bnb = [110.744, 110.886, 126.878, 143.212, 310.814, 119.644, 125.612, 137.858, 149.886, 322.342]

    # Standard deviations for each execution time for each dataset per sampling strategy
    aifb_std_dev = [1.0278, 1.1669, 0.2954, 1.0913, 0.643, 0.5559, 0.9733, 0.8236, 0.7912, 1.5901]
    anime_std_dev = [21.5199, 18.9101, 3.9567, 11.1779, 18.9813, 13.8941, 6.9588, 15.965, 18.9925, 17.409]
    bnb_std_dev = [17.9825, 21.1082, 16.3517, 32.3336, 17.5582, 15.0862, 7.6734, 28.4538, 27.6322, 14.5816]

    sampling_strategies = ['RNS', 'NTS', 'ETS', 'NTETS', 'DBS', 'DCS', 'PRS', 'NTPRS', 'ETPRS', 'NTDBS']

    fig, ax = plt.subplots()

    # Plotting the lines for each dataset
    ax.plot(sampling_strategies, aifb, label='AIFB+')
    ax.plot(sampling_strategies, anime, label='Anime')
    ax.plot(sampling_strategies, bnb, label='BNB')

    # Adding error bars representing the standard deviations
    ax.errorbar(sampling_strategies, aifb, yerr=aifb_std_dev, linestyle='None', color='black', alpha=0.2)
    ax.errorbar(sampling_strategies, anime, yerr=anime_std_dev, linestyle='None', color='black', alpha=0.2)
    ax.errorbar(sampling_strategies, bnb, yerr=bnb_std_dev, linestyle='None', color='black', alpha=0.2)

    # Setting axis labels and title
    ax.set_xlabel('Sampling Strategies')
    ax.set_ylabel('Time (s)')
    ax.set_title('Execution Time for Different Sampling Strategies')

    ax.legend()

    plt.xticks(rotation=45)

    for errorbar in ax.lines[-4:]:
        errorbar.set_alpha(1)

    plt.show()