import dask.dataframe as dd
import time   
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import math

from sklearn.cluster import KMeans
import numpy as np

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
    
    def group_up_str_fields(self, x_2, type_str):
        labels_dct = {}
        for val in x_2:
            if val and (type(val) is str or not math.isnan(val)):
                if type_str == "int":
                    val = int(val)
                if val in labels_dct:
                    labels_dct[val] += 1
                else: 
                    labels_dct[val] = 1
        
        lst_labels = list(labels_dct.keys())
        if type_str == "int":
            lst_labels = sorted(lst_labels)
        lst_data = []
        max_label = None
        max_label_val = 0
        for label in lst_labels:
            temp_val = labels_dct[label]

            lst_data += [temp_val]
            if temp_val > max_label_val:
                max_label_val = temp_val
                max_label = label

        return (lst_labels, lst_data, max_label, max_label_val)

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

    def borough_analysis(self):
        x_2 = self.df["BOROUGH"]
        (lst_boroughs, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, None)

        plt.bar(lst_boroughs, lst_data)
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Boroughs")
        plt.ylabel("Number Reported")
        plt.title("Number of Accidents by Borough")

        plt.show()

    def injuries_analysis(self):
        x_2 = self.df["NUMBER OF PERSONS INJURED"]
        (lst_injuries, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, "int")
        inj_domain = 6
        plt.plot(lst_injuries[:inj_domain], lst_data[:inj_domain])
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Number of Persons Injured per Accident")
        plt.ylabel("Frequency")
        plt.title("Frequency of Number of Persons Injured per Accident")

        plt.show()

    def deaths_analysis(self):
        x_2 = self.df["NUMBER OF PERSONS KILLED"]
        (lst_killed, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, "int")
        death_domain = 4
        plt.plot(lst_killed[:death_domain], lst_data[:death_domain])
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Number of Persons Killed per Accident")
        plt.ylabel("Frequency")
        plt.title("Frequency of Number of Persons Killed")

        plt.show()

    def contributing_factor(self):
        x_2 = self.df["CONTRIBUTING FACTOR VEHICLE 1"]
        (lst_contributing_factors, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, None)
        
        cut_off = 50000
        new_lst_factors = []
        new_lst_data = []
        for i in range(len(lst_contributing_factors)):
            if lst_data[i] > cut_off and lst_contributing_factors[i] != "Unspecified":
                new_lst_factors += [lst_contributing_factors[i]]
                new_lst_data += [lst_data[i]]

        plt.bar([i for i in range(len(new_lst_data))], new_lst_data)
        plt.ylim([0, max_label_val * 0.8])
        plt.xlabel("Contributing Factor")
        plt.ylabel("Frequency")
        plt.title("Frequency of Contributing Factors of Accident")
        
        print(new_lst_factors)
        labels = [f"{i} - {new_lst_factors[i]}" for i in range(len(new_lst_factors))]
        handles = [plt.Rectangle((0,0),1,1, color="red") for label in labels]
        plt.legend(handles, labels) 

        plt.show()

    def k_means_groups(self, lst_facts):
        a = lambda x: 0 if math.isnan(x) else x
        lst_dumb = []
        for val in np.array(self.df[lst_facts].values):
            lst_dumb += [np.array([a(val_2) for val_2 in val])]
        X = np.array(lst_dumb)
        num = len(lst_facts)
        kmeans = KMeans(n_clusters = num).fit(X)
        dct_count = {}
        for label in kmeans.labels_:
            if label in dct_count:
                dct_count[label] += 1
            else:
                dct_count[label] = 1
        print(f"Factors:    {lst_facts}")
        print(f"Labels:     {[ (key, dct_count[key]) for key in dct_count.keys()]}")

        new_centers = []
        for center in kmeans.cluster_centers_:
            new_centers += [[int(val) for val in center]]

        print(f"Centers:     {new_centers}")

    def introductory_analysis(self):
        num_entries = self.df.shape[0].compute()
        print(f"Num motor vehicle accident entries is {num_entries}")
        
        self.day_of_week_analysis()
        self.yearly_analysis()
        self.borough_analysis()
        self.injuries_analysis()
        self.deaths_analysis()
        self.contributing_factor()

        lst_1 = ["NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED", "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED", "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED", "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED"]
        self.k_means_groups(lst_1)


if __name__ == "__main__":
    file_handler = File_Handler()
    file_handler.introductory_analysis()