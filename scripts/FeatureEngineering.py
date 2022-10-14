import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

html_table = pd.DataFrame()
stats_table = pd.DataFrame()


def check_response_type(df, response):
    _isBool = False

    if len(pd.unique(df[response])) == 2:
        _isBool = True

    return _isBool


def check_predictor_type(df, predictor):
    _isCat = True

    if len(pd.unique(df[predictor])) >= 15 and df[predictor].dtype == float:
        _isCat = False

    return _isCat


def plot_bool_response_cat_predictor(df, predictor, response):
    fig = px.density_heatmap(df, x=predictor, y=response)

    fig.show()


def plot_bool_response_con_predictor(df, predictor, response, stats_table):
    group_labels = ["Response = 0", "Response = 1"]

    df = df.dropna()

    group1 = df[df[response] == 0]
    group2 = df[df[response] == 1]

    hist_data = [group1[predictor], group2[predictor]]

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=10)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()

    count, bins_edges = np.histogram(df[predictor], bins=4)

    # print(count)
    # print(bins_edges)
    bins_mean_diff = [0 for i in range(len(bins_edges))]

    print()

    pop_mean = np.mean(df[predictor])
    print("population mean for predictor ", predictor, " = ", pop_mean)

    for i in range(len(bins_edges) - 1):
        bin_data = df[
            (bins_edges[i + 1] >= df[predictor]) & (bins_edges[i] < df[predictor])
        ]
        # print(bin_data[predictor])
        # print()
        bins_mean_diff[i] = np.square(
            count[i] * (np.mean(bin_data[predictor]) - pop_mean)
        )
        print(
            "for predictor ",
            predictor,
            "squared diff of mean for bin ",
            i,
            " is ",
            bins_mean_diff[i],
        )

    print("w mean squared difference = ", np.mean(bins_mean_diff))
    stats_table["w_mean_diff"][predictor] = np.mean(bins_mean_diff)


def logistic_regression(df, predictor_name, response, stats_table):

    x = df[predictor_name]
    y = df[response]

    # Remaking df with just these two columns to remove na's
    df = pd.DataFrame({predictor_name: x, response: y})
    pd.set_option("mode.use_inf_as_na", True)
    df = df.dropna()
    x = df[predictor_name]
    y = df[response].map(int)

    predictor = sm.add_constant(x)
    logistic_regression_model = sm.Logit(np.asarray(y), np.asarray(predictor))
    logistic_regression_model_fitted = logistic_regression_model.fit()
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:6e}".format(logistic_regression_model_fitted.pvalues[1])
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title=f"Variable: {predictor_name}: (t-value={t_value}) (p-value={p_value})",
        xaxis_title=f"Variable: {predictor_name}",
        yaxis_title="y",
    )
    fig.show()
    stats_table["tvalue"][predictor_name] = t_value
    stats_table["pvalue"][predictor_name] = p_value
    print(stats_table)


def main():
    # import data

    global stats_table

    dataframe = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    )

    predictors = ["pclass", "sex", "age", "fare", "embarked"]
    response = "survived"

    dataframe = dataframe[["pclass", "sex", "age", "fare", "embarked", "survived"]]

    # cleaning the dataset
    dataframe.dropna()
    dataframe.reset_index(drop=True)
    dataframe["age"].fillna(dataframe["age"].mean(), inplace=True)
    dataframe["sex"].replace(["male", "female"], [0, 1], inplace=True)
    dataframe["embarked"].replace(["S", "C", "Q"], [0, 1, 2], inplace=True)
    dataframe["embarked"].fillna(dataframe["embarked"].mean(), inplace=True)

    html_table = pd.DataFrame(
        [0.0, 0.0, 0.0, 0.0, 0.0], index=predictors, columns=["dummy"]
    )

    stats_table = pd.DataFrame(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        index=predictors,
        columns=["tvalue", "pvalue", "w_mean_diff"],
    )

    # print(html_table)

    is_bool = check_response_type(dataframe, response)

    for predictor in predictors:

        is_cat = check_predictor_type(dataframe, predictor)

        # plot responses
        if is_cat is True and is_bool is True:
            plot_bool_response_cat_predictor(dataframe, predictor, response)
        if is_cat is False and is_bool is True:
            plot_bool_response_con_predictor(
                dataframe, predictor, response, stats_table
            )

        # perform regression to get t-value ans p-value
        if is_bool is True and is_cat is False:
            logistic_regression(dataframe, predictor, response, stats_table)

    # separate data and target for random forest
    X = dataframe.drop("survived", axis=1)
    y = dataframe["survived"]

    rf = RandomForestClassifier()
    rf.fit(X, y)

    feature_importance = pd.DataFrame(
        rf.feature_importances_, index=X.columns, columns=["importance"]
    ).sort_values("importance", ascending=True)
    # print(feature_importance)

    html_table = pd.concat([html_table, feature_importance, stats_table], axis=1)
    html = html_table.to_html()

    # write html to file
    text_file = open("index.html", "w")
    text_file.write(html)
    text_file.close()


if __name__ == "__main__":
    main()
