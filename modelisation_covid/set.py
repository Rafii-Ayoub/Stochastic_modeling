import pandas as pd
from functions import *


class Set:

    def __init__(self, dataset):
        self.dataset = dataset

    def select_country(self,country):
        return self.dataset[self.dataset["Country/Region"] == country]

    def select_date(self,date):
        return self.dataset[self.dataset["Date"] == date]

    def select_province(self,province):
        return self.dataset[self.dataset["Province/State"] == province]

    def select_date_interval(self,start_date, end_date):
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        mask = (self.dataset['Date'] > start_date) & (self.dataset['Date'] <= end_date)
        return self.dataset.loc[mask]


    def select_country_from_to(self,country,start_date, end_date):
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'] )
        mask = (self.dataset['Date'] > start_date) & (self.dataset['Date'] <= end_date)
        return self.dataset.loc[mask][self.dataset["Country/Region"] == country]