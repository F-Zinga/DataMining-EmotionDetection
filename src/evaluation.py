from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, f1_score
import scikitplot as skplt
import matplotlib.pyplot as plt


def evaluate(model, x_test, y_test):

    grid_predictions = model.predict(x_test)

    # print classification report
    print(classification_report(y_test, grid_predictions))
    rfc = model.best_estimator_
    print("\nC parameter's value used: ")
    print(rfc)
    print("\nAUC: ")
    print(roc_auc_score(y_test, grid_predictions))
    print("\nF1: ")
    print(f1_score(y_test, grid_predictions))
    skplt.metrics.plot_confusion_matrix(grid_predictions, y_test)
    plt.show()
    return model.score(x_test, y_test)
