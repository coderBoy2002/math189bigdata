import dask.dataframe as dd
import time   
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt

class File_Handler(object):
    def __init__(self):
        self.file_path = file_path = "data/Motor_Vehicle_Accidents_NY.parquet"
        start_time = time.time()
        self.df = dd.read_parquet(self.file_path, engine='pyarrow')
        print(f"Loading data took {round(time.time() - start_time,2)} (s)")
        
        self.latest_date = None
        self.earliest_date = None
        pass

    def func1(self, y):
        d1 = y.split("/")
        date_1 = datetime(int(d1[2]), int(d1[0]), int(d1[1]))

        if self.latest_date and self.earliest_date:
            if self.latest_date < date_1:
                self.latest_date = date_1
            elif self.earliest_date > date_1:
                self.earliest_date = date_1
        else:
            self.latest_date = date_1
            self.earliest_date = date_1
        return date_1.isoweekday()

    def func2(self, y):
        d1 = y.split("/")
        return int(d1[2])

    def day_of_week_analysis(self):
        x_1 = self.df["CRASH DATE"].map(self.func1, meta=("CRASH DATE", str))
        days_of_week_data = [0]*7
        days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for val in x_1:
            days_of_week_data[val - 1] +=1

        y_max = max(days_of_week_data)
        plt.plot(days_of_week, days_of_week_data)
        plt.ylim([0, y_max * 1.2])
        plt.xlabel("Day of the Week")
        plt.ylabel("Number Reported")
        plt.title("Number of Accidents by the Day of the Week")
        plt.show()

    def yearly_analysis(self):
        print(f"Earliest date: {self.earliest_date}")
        print(f"Latest date: {self.latest_date}")


        x_2 = self.df["CRASH DATE"].map(self.func2, meta=("CRASH DATE", str))

        years_data = [0]*13
        years = [2012, 2013, 2014, 2015, 2016, 2017, 2018,
                 2019, 2020, 2021, 2022, 2023, 2024]

        for val in x_2:
            years_data[val - 2012] += 1
        
        y_max_2 = max(years_data)

        plt.plot(years, years_data)
        plt.ylim([0, y_max_2 * 1.2])
        plt.xlabel("Year")
        plt.ylabel("Number Reported")
        plt.title("Number of Accidents by Year")

        plt.show()

    def introductory_analysis(self):
        num_entries = self.df.shape[0].compute()
        print(f"Num motor vehicle accident entries is {num_entries}")
        
        self.day_of_week_analysis()
        self.yearly_analysis()


    




if __name__ == "__main__":
    file_handler = File_Handler()
    file_handler.introductory_analysis()