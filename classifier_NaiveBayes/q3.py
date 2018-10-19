import pandas as pd
import numpy as np

"""
To run this code, make sure the data-sets are located in data/ directory 
in the same folder.

"""


# -----------------------------------------------------------------------------
def main():
    """
    """
    # Training the classifier
    training_data = pd.read_csv("data/q3.csv")

    # Filter data set wrt Spam
    spam = training_data.values[np.where(training_data.values[:, -1] == True)]
    not_spam = training_data.values[
                                np.where(training_data.values[:, -1] == False)]

    # Parameters to compute: p(x | Spam), p(x | !Spam); where x is all features

    # Calculate prior probability of spam and not spam
    p_spam = len(spam) / len(training_data)
    p_not_spam = 1 - p_spam

    # Feature parameter lists for spam and not spam classes
    spam_parameters = [(0, 0) for _ in range(8)]
    not_spam_parameters = [(0, 0) for _ in range(8)]

    # Bernoulli parameters
    for feature_index in range(6):
        spam_parameters[feature_index], not_spam_parameters[feature_index] = \
                        get_bernoulli_parameter(spam, not_spam, feature_index)
    # Gaussian parameters
    for feature_index in range(6, 8):
        spam_parameters[feature_index], not_spam_parameters[feature_index] = \
                        get_gaussian_parameters(spam, not_spam, feature_index)

    # Feature names for pretty printing
    features = ["in html", "has emoji", "sent to list", "from .com", 
                "has my name", "has sig", "# sentences", "# words"]

    # Print all parameters for all features for q3.a
    print("Q3 a)\nBernoulli parameters: p")
    for i, feature in enumerate(features):  
        if i == 6:
            print("\nGaussian Parameters: mean, variance")

        if i < 6:
            print('{}:\n\tSpam: {},\n\tNot_Spam: {}'.format(
                feature, '%.2f' % spam_parameters[i],
                    '%.2f' % not_spam_parameters[i]))
        else:
            print('{}:\n\tSpam: {},\n\tNot_Spam: {}'.format(
                feature, '%.2f, %.2f' % spam_parameters[i],
                         '%.2f, %.2f' % not_spam_parameters[i]))

    # Testing classifier error
    test_data = pd.read_csv("data/q3b.csv").values

    # Combination of feature-indices to be used for prediction
    feature_list = [3, 4, 5, 6, 7]

    # Predict class for each sample in test set, and compute error
    error_count = 0
    for data_point in test_data:
        verdict = classify(data_point, feature_list, spam_parameters,
                           not_spam_parameters, p_spam, p_not_spam)
        if verdict != data_point[-1]:
            error_count += 1

    classifier_error = error_count / len(test_data)

    print("\nQ3 b) and c)")
    print("Classifier error with features {}: {}".format(
                                    list(np.array(features)[feature_list]),
                                    classifier_error))
# ------------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def classify(data_point, feature_list, spam_parameters, not_spam_parameters,
                                                            p_spam, p_not_spam):
    """

    The class (Spam or not-Spam) that is most likely
    has greater joint probability.

    P(Spam | everything else) =
                    Multiply_for_all_f(P(f | Spam) * P(Spam)) / evidence
                    ; evidence can be ignored as it is constant in both Spam and 
                                                        notSpam probabilities
                = Multiply_for_all_f(P(f | Spam)) * P(Spam)

    P(Not Spam | everything else) =
                    Multiply_for_all_f(P(f | Not Spam) * P(Not Spam)) / evidence
                    ; evidence can be ignored as it is constant in both Spam and
                                                        notSpam probabilities
                = Multiply_for_all_f(P(f | Not Spam)) * P(Not Spam)

    Take logs before multiplying for efficiency
        log_total_spam = Sum(ln(P(f | Spam))) + ln(P(Spam))
        log_total_not_spam = Sum(ln(P(f | Not Spam))) + ln(P(Not Spam))

    The larger of these two values is our prediction.

    @:param data_point: sample data
    @:param feature_list: list of feature indices to be used for prediction
    @:param spam_parameters: bernoulli parameters for boolean features,
                            and gaussian parameters for real-valued features
                            for distribution spam = True
    @:param not_spam_parameters: bernoulli parameters for boolean features,
                            and gaussian parameters for real-valued features
                            for distribution spam = False
    @:param p_spam: Prior probability of spam
    @:param p_not_spam: Prior probability of not spam

    """
    # Init sum buffers
    log_theta_spam = np.log(p_spam)
    log_theta_not_spam = np.log(p_not_spam)

    for feature_index in feature_list:

        # Calc posterior probability
        if feature_index < 6:
            if data_point[feature_index]:
                p = spam_parameters[feature_index]
                q = not_spam_parameters[feature_index]
            else:
                p = 1 - spam_parameters[feature_index]
                q = 1 - not_spam_parameters[feature_index]
        else:
            p = gaussian(spam_parameters[feature_index][0],
                         spam_parameters[feature_index][1],
                         data_point[feature_index])
            q = gaussian(not_spam_parameters[feature_index][0],
                         not_spam_parameters[feature_index][1],
                         data_point[feature_index])

        log_theta_spam += np.log(p)
        log_theta_not_spam += np.log(q)

    return log_theta_spam > log_theta_not_spam
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def get_bernoulli_parameter(spam, not_spam, feature_index):
    """
    @:param spam: data-set that has spam = True
    @:param not_spam: data-set that has spam = False
    @:param feature_index: Feature index whose bernoulli parameters
                            are to be computed.
    """
    
    feature_true = spam[np.where(spam[:, feature_index] == True)]
    p1 = len(feature_true)/len(spam)

    feature_false = not_spam[np.where(not_spam[:, feature_index] == True)]
    p2 = len(feature_false)/len(not_spam)

    return p1, p2
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def get_gaussian_parameters(spam, not_spam, feature_index):
    """
    @:param spam: data-set that has spam = True
    @:param not_spam: data-set that has spam = False
    @:param feature_index: Feature index whose gaussian parameters
                            are to be computed.

    """
    m1 = np.mean(spam[:, feature_index])
    v1 = np.var(spam[:, feature_index])

    m2 = np.mean(not_spam[:, feature_index])
    v2 = np.var(not_spam[:, feature_index])

    return (m1, v1), (m2, v2)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
def gaussian(m, v, x):
    """
    Returns gaussian curve value with mean = m, variance = v, at point x
    """
    
    normalizer = 1/np.sqrt(2 * np.pi * v)
    exp_term = -np.square(x - m)/(2 * v)

    return normalizer * (np.e ** exp_term)
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
