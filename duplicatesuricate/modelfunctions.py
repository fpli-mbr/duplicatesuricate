import pandas as pd


def model_train(model,training_set=None,verbose=True):







        print('score on training data', score)
    return model,traincols


def model_add(used_model=None, warmstart=False, training_set=None, n_estimators=2000, traincols=None):
    """
    this function initiate and fits the model on the specified training table
    Args:
        training_set (pd.DataFrame): supervised learning training table, has ismatch column
        n_estimators(int): number of estimators used for standard RandomForestModel
        used_model (scikit-learn Model): model used to do the prediction, default RandomForest model
        warmstart (bool): say wheter to train the model or not


    Returns:

    """

    # define the model
    if used_model is None:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators)
    else:
        model = used_model
        traincols = traincols

    if warmstart is False:
        if training_set is None:
            raise ('Error no training set provided')
        else:
            model,traincols=model_train(training_set=training_set)
    return model,traincols