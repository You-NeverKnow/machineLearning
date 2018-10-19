import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#------------------------------------------------------------------------------
def main():
    """
    """
    filenames = ["Frogs-subsample.csv", "Frogs.csv"]
    for filename in filenames:
        frogData = pd.read_csv("data/" + filename).values 

        # Features
        x1 = frogData[np.where(frogData[:,2] == "HylaMinuta")][:,0]
        y1 = frogData[np.where(frogData[:,2] == "HylaMinuta")][:,1]
        
        x2 = frogData[np.where(frogData[:,2] != "HylaMinuta")][:,0]
        y2 = frogData[np.where(frogData[:,2] != "HylaMinuta")][:,1]

        #------------------#
        ## Visualizations ##
        #------------------#

        # Draw scatter plots
        create_scatter_plot(x1, y1, x2, y2, filename)

        # Draw histograms
        create_histograms(x1, y1, x2, y2, filename)

        # Draw line-graphs
        create_line_graphs(x1, y1, x2, y2, filename)

        # Box-plots
        create_box_plot(x1, y1, x2, y2, filename)        

        # Error-plots    
        create_error_bars(x1, y1, x2, y2, filename)

        #--------------#
        ## Statistics ##
        #--------------#
        
        # Mean
        mean_x1 = np.mean(x1)
        mean_y1 = np.mean(y1)
        mean_x2 = np.mean(x2)
        mean_y2 = np.mean(y2)
        
        # Co-variance matrix
        covariance_matrix_1 = np.cov(list(x1), 
                                    list(y1), bias=True)
        covariance_matrix_2 = np.cov(list(x2), 
                                    list(y2), bias=True)
        
        # Standard deviation
        std_x1 = np.std(x1)
        std_x2 = np.std(x2)
        
        std_y1 = np.std(y1)
        std_y2 = np.std(y2)

        precision_factor = '%.4f'

        # Save stats
        with open("statistics/stats_" + filename, "w") as f:
            f.write("mean_x1, mean_y1 = " + precision_factor % mean_x1 + \
                            ", " + precision_factor %  mean_y1 + "\n")
            f.write("std_x1, std_y1 = " + precision_factor % std_x1 + \
                            ", " + precision_factor % std_y1 + "\n")
            f.write("Covariance matrix_1 = " + "\n")
            f.write("\t" + precision_factor % covariance_matrix_1[0][0] + \
                    ", " + precision_factor % covariance_matrix_1[0][1] + "\n")
            f.write("\t" + precision_factor % covariance_matrix_1[1][0] + \
                    ", " + precision_factor % covariance_matrix_1[1][1] + "\n\n")


            f.write("mean_x2, mean_y2 = " + precision_factor % mean_x2 + \
                ", " + precision_factor %  mean_y2 + "\n")
            f.write("std_x2, std_y2 = " + precision_factor % std_x2 + \
                            ", " + precision_factor % std_y2 + "\n")
            f.write("Covariance matrix_2 = " + "\n")
            f.write("\t" + precision_factor % covariance_matrix_2[0][0] + \
                    ", " + precision_factor % covariance_matrix_2[0][1] + "\n")
            f.write("\t" + precision_factor % covariance_matrix_2[1][0] + \
                    ", " + precision_factor % covariance_matrix_2[1][1] + "\n")
        
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def create_scatter_plot(x1, y1, x2, y2, filename):
    """
    """

    sample_name = "" if filename == "Frogs.csv" else "(sub)" 
    
    plt.scatter(x1, y1, color="royalblue", label="HylaMinuta")
    plt.scatter(x2, y2, color="green", 
                        label="HypsiboasCinerascens", alpha=0.7)
    
    plt.ylim([-.5,.5])
    plt.xlim([-.6,.6])

    plt.xlabel("MFCC_10")
    plt.ylabel("MFCC_17")
    plt.legend(("HylaMinuta", "HypsiboasCinerascens"))
    plt.title("Sound Frequency-band-intensities" + sample_name)

    plt.savefig('plots/scatter_' + filename + '.png', bbox_inches='tight')
    plt.clf()

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def create_histograms(x1, y1, x2, y2, filename):
    """
    """
    features = [[x1, y1], [x2, y2]]
    frog_names = ["HylaMinuta", "HypsiboasCinerascens"]
    sample_name = "" if filename == "Frogs.csv" else "(sub)" 

    labels = ("MFCC_10", "MFCC_17")
    
    for j in range(2):
        plt.title("Frequency-band-intensity Distribution: " + \
                                                frog_names[j] + sample_name)
        for feature in features[j]:
            plt.hist(sorted(feature), bins = len(feature), label=labels[j])

        plt.xlabel("Frequency-band-intensities(MFCC)")
        plt.ylabel("Frequency of MFCC")
        plt.ylim([0, 10])
        plt.xlim([-.6,.6])
        plt.legend(("MFCC_10", "MFCC_17"))
        plt.savefig('plots/histogram_' + filename + '_' + frog_names[j] + \
                                            '.png', bbox_inches='tight')
        plt.clf()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def create_line_graphs(x1, y1, x2, y2, filename):
    """
    """
    features = [[x1, y1], [x2, y2]]
    frog_names = ["HylaMinuta", "HypsiboasCinerascens"]
    labels = ("MFCC_10", "MFCC_17")
    sample_name = "" if filename == "Frogs.csv" else "(sub)" 
    
    for j in range(2):

        for i, feature in enumerate(features[j]):
            plt.plot(np.linspace(0, len(feature), len(feature)), 
                                        sorted(feature), label=labels[i])

        plt.title("Frequency-band-intensity Distribution curve: " + \
                                                frog_names[j] + sample_name)
        plt.xlabel("Coefficient units")
        plt.ylabel("Frequency band intensities")
        plt.ylim([-0.6, 0.6])

        if filename == "Frogs.csv":
            plt.xlim([-20, 530])    
        else:
            plt.xlim([-2, 30])
        plt.legend(("MFCC_10", "MFCC_17"))

        plt.savefig('plots/lineplot_' + filename + '_' + frog_names[j] + \
                                            '.png', bbox_inches='tight')
        plt.clf()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def create_box_plot(x1, y1, x2, y2, filename):
    """
    """

    sample_name = "" if filename == "Frogs.csv" else "(sub)" 
    plt.title("Distribution of MFCCs of frogs"+ sample_name)
    plt.xlabel("Two MFCCs of two frogs")
    plt.ylabel("Frequency band intensities")
    plt.ylim([-0.6, 0.6])
    plt.xlim([0, 4])
    
    plt.boxplot([x1, x2, y1, y2])

    plt.xticks([1, 2, 3, 4], ['a', 'b', 'c', 'd'])
    plt.figtext(0.9, 0.8, "a: HylaMinuta- MFCC_10", 
                            backgroundcolor="royalblue", color="darkkhaki")
    plt.figtext(0.9, 0.75, "b: HypsiboasCinerascens- MFCC_10", 
                            backgroundcolor="royalblue", color="darkkhaki")
    plt.figtext(0.9, 0.70, "c: HylaMinuta- MFCC_17",
                            backgroundcolor="royalblue", color="darkkhaki")
    plt.figtext(0.9, 0.65, "d: HypsiboasCinerascens- MFCC_17", 
                            backgroundcolor="royalblue", color="darkkhaki")

    plt.savefig('plots/box_' + filename + '.png', bbox_inches='tight')
    plt.clf()

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def create_error_bars(x1, y1, x2, y2, filename):
    """
    """
    sample_name = "" if filename == "Frogs.csv" else "(sub)" 

    means = [np.mean(x1), np.mean(x2), np.mean(y1), np.mean(y2)]
    stds = [np.std(x1), np.std(x2), np.std(y1), np.std(y2)]

    plt.title("Error in MFCCs" + sample_name)
    plt.xlabel("Two MFCCs of two frogs")
    plt.ylabel("Mean of Frequency band intensities")
    plt.ylim([-0.4, 0.4])
    plt.xlim([-0.1, 5])

    plt.xticks([1, 2, 3, 4], ['a', 'b', 'c', 'd'])
    plt.figtext(0.9, 0.8, "a: HylaMinuta- MFCC_10", 
                            backgroundcolor="royalblue", color="darkkhaki")
    plt.figtext(0.9, 0.75, "b: HypsiboasCinerascens- MFCC_10", 
                            backgroundcolor="royalblue", color="darkkhaki")
    plt.figtext(0.9, 0.70, "c: HylaMinuta- MFCC_17",
                            backgroundcolor="royalblue", color="darkkhaki")
    plt.figtext(0.9, 0.65, "d: HypsiboasCinerascens- MFCC_17", 
                            backgroundcolor="royalblue", color="darkkhaki")

    plt.bar(range(1,5), means, yerr = stds, ecolor="red")
    plt.savefig('plots/error_' + filename + '.png', bbox_inches='tight')
    plt.clf()

#------------------------------------------------------------------------------



if __name__ == "__main__":
    main()