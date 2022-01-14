import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import set
import numpy as np
from functions import *


class Plot:

    def __init__(self, dataset):
        self.dataset = dataset

    def draw_cumulated_cases(self, country):
        x = self.dataset.select_country(country)["Date"]
        y = self.dataset.select_country(country)["Confirmed"]
        title = " Covid Cumulated cases in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def draw_cumulated_deaths(self, country):
        x = self.dataset.select_country(country)["Date"]
        y = self.dataset.select_country(country)["Deaths"]
        title = " Covid Cumulated deaths in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def draw_cumulated_recovered(self, country):
        x = self.dataset.select_country(country)["Date"]
        y = self.dataset.select_country(country)["Recovered"]
        title = " Covid Cumulated recovered in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def draw_daily_cases(self, country):

        x = self.dataset.select_country(country)["Date"]
        list = self.dataset.select_country(country)["Confirmed"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        title = " Covid Daily cases in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def draw_daily_deaths(self, country):

        x = self.dataset.select_country(country)["Date"]
        list = self.dataset.select_country(country)["Deaths"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        title = " Covid Daily deaths in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def draw_daily_recovered(self, country):

        x = self.dataset.select_country(country)["Date"]
        list = self.dataset.select_country(country)["Recovered"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        title = " Covid Daily recovered in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()


    def draw_daily_cases_from_to(self, country, start_date, end_date):

        x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
        list = self.dataset.select_country_from_to(country, start_date, end_date)["Confirmed"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        title = " Covid Daily cases in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def draw_log_cumulated_cases_from_to(self, country, start_date, end_date):

        x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
        y = np.log10(np.abs(self.dataset.select_country_from_to(country, start_date, end_date)["Confirmed"].to_numpy()))

        title = " Covid logarithmic cumulated cases in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def draw_cumulated_recovered_from_to(self, countries, start_date, end_date):

        title = " Covid cumulated recovered in : "
        for country in countries:
            x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
            y = self.dataset.select_country_from_to(country, start_date, end_date)["Recovered"].to_numpy()
            title = title + str(country) +"  ; "
            plt.plot(x, y)
            plt.legend( country)
        plt.title(title)
        plt.show()

    def draw_cumulated_deaths_from_to(self, countries, start_date, end_date):

        title = " Covid cumulated deaths in : "
        for country in countries:
            x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
            y = self.dataset.select_country_from_to(country, start_date, end_date)["Deaths"].to_numpy()
            title = title + str(country) +"  ; "
            plt.plot(x, y)
            plt.legend( country)
        plt.title(title)
        plt.show()

    def draw_cumulated_cases_from_to(self, countries, start_date, end_date):

        title = " Covid logarithmic cumulated cases in : "
        for country in countries:
            x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
            y = self.dataset.select_country_from_to(country, start_date, end_date)["Confirmed"].to_numpy()
            title = title + str(country) +"  ; "
            plt.plot(x, y)
            plt.legend( country)
        plt.title(title)
        plt.show()

    def draw_daily_deaths_from_to(self, country, start_date, end_date):

        x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
        list = self.dataset.select_country_from_to(country, start_date, end_date)["Deaths"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        title = " Covid Daily deaths in " + str(country)
        plt.plot(x, y)
        plt.title(title)
        plt.show()

    def draw_daily_cases_with_moving_average(self, country, ma_coeff):
        mac = ma_coeff
        x = self.dataset.select_country(country)["Date"]
        list = self.dataset.select_country(country)["Confirmed"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        z = moving_average(y, mac)
        n = len(z)
        mac = mac // 2
        xx = x[mac:n + mac]
        title = " Covid Daily cases in " + str(country)
        plt.plot(x, y)
        plt.plot(xx, z)
        plt.title(title)
        plt.show()

    def draw_daily_deaths_with_moving_average(self, country, ma_coeff):
        mac = ma_coeff
        x = self.dataset.select_country(country)["Date"]
        list = self.dataset.select_country(country)["Deaths"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        z = moving_average(y, mac)
        n = len(z)
        mac = mac // 2
        xx = x[mac:n + mac]
        title = " Covid Daily deaths in " + str(country)
        plt.plot(x, y)
        plt.plot(xx, z)
        plt.title(title)
        plt.show()

    def draw_daily_recovered_with_moving_average(self, country, ma_coeff):
        mac = ma_coeff
        x = self.dataset.select_country(country)["Date"]
        list = self.dataset.select_country(country)["Recovered"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        z = moving_average(y, mac)
        n = len(z)
        mac = mac // 2
        xx = x[mac:n + mac]
        title = " Covid Daily recovered in " + str(country)
        plt.plot(x, y)
        plt.plot(xx, z)
        plt.title(title)
        plt.show()

    def draw_daily_cases_from_to_with_moving_average(self, country, start_date, end_date, ma_coeff):
        mac = ma_coeff
        x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
        list = self.dataset.select_country_from_to(country, start_date, end_date)["Confirmed"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        title = " Covid Daily cases in " + str(country)
        z = moving_average(y, mac)
        n = len(z)
        mac = mac // 2
        xx = x[mac:n + mac]
        rolling_std = self.dataset.select_country_from_to(country, start_date, end_date)["Confirmed"].rolling(
            window=12).std()
        max = rolling_std.max()
        rolling_std = rolling_std/10
        plt.plot(x, y)
        plt.plot(xx, z)
        plt.plot(x, rolling_std)
        plt.title(title)
        plt.show()

    def draw_daily_deaths_from_to_with_moving_average(self, country, start_date, end_date, ma_coeff):
        mac = ma_coeff
        x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
        list = self.dataset.select_country_from_to(country, start_date, end_date)["Death"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        title = " Covid Daily deaths in " + str(country)
        z = moving_average(y, mac)
        n = len(z)
        mac = mac // 2
        xx = x[mac:n + mac]
        plt.plot(x, y)
        plt.plot(xx, z)
        plt.title(title)
        plt.show()

    def draw_daily_recovered_from_to_with_moving_average(self, country, start_date, end_date, ma_coeff):
        mac = ma_coeff
        x = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]
        list = self.dataset.select_country_from_to(country, start_date, end_date)["Recovered"].to_numpy()
        y = np.zeros(len(list))
        for k in range(len(list) - 1):
            y[k + 1] = list[k + 1] - list[k]
        title = " Covid Daily recovered in " + str(country)
        z = moving_average(y, mac)
        n = len(z)
        mac = mac // 2
        xx = x[mac:n + mac]
        t = rolling_window(y, mac)
        print(t)

        plt.plot(x, y)
        plt.plot(xx, z)

        plt.title(title)
        plt.show()

    def draw_correlation_dataset_cases(self):
        old = self.dataset
        new = pd.DataFrame(old[["Infected cases", "Total Tests Conducted","Pop. Density (per km2)", "Avg. Annual Temp. (C.)","%Pop. Poor"]])
        df_fill = new.fillna("")
        df_num = df_fill.apply(lambda x: pd.factorize(x)[0]) + 1
        corr = df_num.corr()
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );
        ax.set_title("Correlation Analyses 1 (Cases)")
        plt.show()


    def draw_correlation_dataset_deaths(self):
        old = self.dataset
        new = pd.DataFrame(old[["Deaths", "Infected cases", "Serious cases","%Pop. High Income","%Pop. Low Income","%Pop. Age 15-64" ,"%Pop. Age >=65"]])
        df_fill = new.fillna("")
        df_num = df_fill.apply(lambda x: pd.factorize(x)[0]) + 1
        corr = df_num.corr()
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True,

        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right',

        );
        ax.set_title("Correlation Analyses 2 (Deaths)")
        plt.show()

    def draw_entropy(self, country, start_date, end_date):
        matrix = self.dataset.select_country_from_to(country, start_date, end_date)["Confirmed"].to_numpy()
        date = self.dataset.select_country_from_to(country, start_date, end_date)["Date"]

        x = entropy_local(10, matrix)

        plt.title("Locale entropie")
        plt.xlabel("date")
        plt.ylabel("max entropy value")
        plt.plot(date[10:len(date) - 10], x)
        plt.show()





