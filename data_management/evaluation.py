import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


# Also add the mean accuracy of the cross-validation splits
def get_best_cv_scores(grid_search_clf):
    cv_results = grid_search_clf.cv_results_
    cv_scores = {}
    for i in range(grid_search_clf.n_splits_):
        cv_scores[f'Fold {i + 1}'] = {
            'Test Accuracy': cv_results[f'split{i}_test_score'][grid_search_clf.best_index_],
            'Train Accuracy': cv_results[f'split{i}_train_score'][
                grid_search_clf.best_index_]}
    cv_scores['Mean'] = {
        'Test Accuracy': cv_results['mean_test_score'][grid_search_clf.best_index_],
        'Train Accuracy': cv_results['mean_train_score'][grid_search_clf.best_index_]}

    return pd.DataFrame.from_dict(cv_scores, orient='index')


def estimate_accuracy(grid_search_clf):
    cv_results = get_best_cv_scores(grid_search_clf)[:grid_search_clf.n_splits_]
    indices = cv_results.index.map(lambda x: int(x.split(' ')[1]))

    # Estimate the accuracy of the next fold using a linear regression and plot the trend line and the prediction
    x = indices.to_numpy().reshape(-1, 1)
    y = cv_results['Test Accuracy'].to_numpy().reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    x_pred = np.array([[0]])
    y_pred = reg.predict(x_pred)

    x_plot = np.append(x, x_pred)
    y_plot = np.append(reg.predict(x), y_pred)
    sort_indices = np.argsort(x_plot, axis=0).flatten()
    x_plot = x_plot[sort_indices]
    y_plot = y_plot[sort_indices]

    # Place the trend line behind the scatter plots
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_plot, linestyle='--', label='Test Accuracy Trend', color='C1', zorder=1)
    plt.scatter(indices, cv_results['Test Accuracy'], label='Test Accuracy', s=75, color='C1', zorder=2)
    plt.scatter(x_pred, y_pred, label='Accuracy Prediction', s=75, color='C2', zorder=2)
    plt.scatter(indices, cv_results['Train Accuracy'], label='Train Accuracy', s=75, color='C0', zorder=2)

    plt.title('Cross-Validation Scores')
    plt.xlabel('Number of Removed Orders (per User)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.show()

    return y_pred[0][0]


def eval_logreg(grid_search_clf, features):
    best_estimator = grid_search_clf.best_estimator_
    logreg_model = best_estimator.named_steps['logreg']
    coefficients = logreg_model.coef_
    print("Coefficients:", coefficients)

    plt.figure(figsize=(10, 6))
    plt.bar(features, coefficients[0])
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    plt.title('Coefficients of Linear Model')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
