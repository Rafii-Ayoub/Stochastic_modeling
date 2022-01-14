import pandas as pd
import set
import plot
import analyses
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


"""------------Analyses 1st dataset ---------------"""

df = pd.read_csv("covid_data.csv")
print(df.columns)

df = set.Set(df)
data_for_plot = plot.Plot(df)
dfp = data_for_plot

"""  Analyses: Covid cases - deaths - recovered """


countries=["Germany","Spain","Morocco","Brazil"]
#dfp.draw_cumulated_deaths_from_to(countries, "2020/01/07" ,"2021/01/8")
#print(df.select_country_from_to("Spain","2021/01/07" ,"2022/01/8"))

#dfp.draw_daily_cases_from_to("Germany","2020/03/07" ,"2021/01/8")
#dfp. draw_daily_cases_from_to_with_moving_average("Germany","2021/01/07" ,"2022/01/8",20)

"""  Analyses: evolution characteristics - change detection  """

#dfp.draw_log_cumulted_cases_from_to("Germany","2021/01/07" ,"2022/01/8")
#dfp.draw_entropy("Germany","2020/03/07" ,"2022/01/8")


ad = analyses.analyses(df)
#print(ad.covariance_deaths(["Spain","Morocco","Senegal"]))

"""------------Analyses 2nd dataset ---------------"""

df2 = pd.read_csv('covid_dataset2.csv')
#print(df2.columns)
dfp = plot.Plot(df2)
dfp2 = set.Set(dfp)

"""  Correlation analyses of the 2nd dataset """

#dfp.draw_correlation_dataset_cases()
#dfp.draw_correlation_dataset_deaths()

