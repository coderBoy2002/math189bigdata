import dask.dataframe as dd
import time   
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
import math
import requests
import pandas as pd

from sklearn.cluster import KMeans
import numpy as np
from transformers import AutoTokenizer
import dask.array as da

from sentence_transformers import SentenceTransformer, util
import csv

api_token = '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'
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return list(response.json())


class File_Handler(object):
    def __init__(self):
        self.file_path = file_path = "data/Motor_Vehicle_Accidents_NY.parquet"
        start_time = time.time()
        self.df = dd.read_parquet(self.file_path, engine='pyarrow')
        print("Finished Loading")
        self.df["CRASH TIME"] =  self.df["CRASH TIME"].apply(lambda x: x.split(":")[0])
        
        self.cause_data_lst = []
        with open('./data/CLEAN_Cause_Data.csv', mode ='r') as file:    
            csvFile = csv.DictReader(file)
            for lines in csvFile:
                self.cause_data_lst += [lines['0']]
        '''
            CLEAN UP NLP
        start_time = time.time()
        source_sentences = ["Unspecified", "Aggressive Driving", "Driver Inattention/Distraction", "Alcohol Involvement"]
        
        #num_batches = 272
        num_batches = 100
        batch_size = 300
        clean_contributing_factor = []
        start_time = time.time()

        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        for i in range(num_batches):
            if i%2 == 0:
                print(f"{i}     time: {round(time.time() - start_time, 2)}")
                start_time = time.time()
            clean_none = lambda x: x if x else "Unspecified"
            batch = clean_none(list(self.df["CONTRIBUTING FACTOR VEHICLE 1"][i * batch_size:(i + 1) *  batch_size]))
            scores = []
            for source_sentence in source_sentences:
                query = source_sentence
                docs = batch

                #Encode query and documents
                query_emb = model.encode(query)
                doc_emb = model.encode(docs)

                data = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

                scores += [data]
            clean_lst = []
            for j in range(len(scores[0])):
                if batch[j] == "Unspecified":
                    clean_lst += ["Unspecified"]
                else:
                    temp_scores = [score[j] for score in scores]
                    best_score = max(temp_scores)
                    index_score = temp_scores.index(best_score)
                    clean_lst += [source_sentences[index_score]]

            clean_contributing_factor += clean_lst

        #df = pd.DataFrame(np.array(clean_contributing_factor))
        #df.to_csv("./data/CLEAN_Cause_Data.csv")
            '''

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

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        field_name = "day_of_week"
        plt.savefig(f'./graphs/{field_name}.png')
        plt.close()

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

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        field_name = "yearly"
        plt.savefig(f'./graphs/{field_name}.png')
        plt.close()

    def borough_analysis(self):
        x_2 = self.df["BOROUGH"]
        (lst_boroughs, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, None)

        plt.bar(lst_boroughs, lst_data)
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Boroughs")
        plt.ylabel("Number Reported")
        plt.title("Number of Accidents by Borough")

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        field_name = "borough"
        plt.savefig(f'./graphs/{field_name}.png')
        plt.close()

    def crashtime_analysis(self):
        x_2 = self.df["CRASH TIME"]
        (lst_boroughs, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, "int") 

        plt.plot(lst_boroughs, lst_data)
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Hour")
        plt.ylabel("Number Reported")
        plt.title("Number of Accidents by Crash Time")

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        field_name = "crash_time"
        plt.savefig(f'./graphs/{field_name}.png')
        plt.close()

    def injuries_analysis(self):
        x_2 = self.df["NUMBER OF PERSONS INJURED"]
        (lst_injuries, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, "int")
        inj_domain = 6
        plt.plot(lst_injuries[:inj_domain], lst_data[:inj_domain])
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Number of Persons Injured per Accident")
        plt.ylabel("Frequency")
        plt.title("Frequency of Number of Persons Injured per Accident")

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        field_name = "injuries"
        plt.savefig(f'./graphs/{field_name}.png')
        plt.close()

    def deaths_analysis(self):
        x_2 = self.df["NUMBER OF PERSONS KILLED"]
        (lst_killed, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, "int")
        death_domain = 4
        plt.plot(lst_killed[:death_domain], lst_data[:death_domain])
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Number of Persons Killed per Accident")
        plt.ylabel("Frequency")
        plt.title("Frequency of Number of Persons Killed")

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        field_name = "deaths"
        plt.savefig(f'./graphs/{field_name}.png')
        plt.close()

    def contributing_factor(self):
        x_2 = self.cause_data_lst
        (lst_contributing_factors, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, None)
        
        cut_off = 1000
        new_lst_factors = []
        new_lst_data = []
        for i in range(len(lst_contributing_factors)):
            if lst_data[i] > cut_off and lst_contributing_factors[i] not in ["Unspecified","Unspecificed"]:
                new_lst_factors += [lst_contributing_factors[i]]
                new_lst_data += [lst_data[i]]

        plt.bar([i for i in range(len(new_lst_data))], new_lst_data)
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Contributing Factor")
        plt.ylabel("Frequency")
        plt.title("Frequency of Contributing Factors of Accident")
        
        print(new_lst_factors)
        labels = [f"{i} - {new_lst_factors[i]}" for i in range(len(new_lst_factors))]
        handles = [plt.Rectangle((0,0),1,1, color="red") for label in labels]
        plt.legend(handles, labels) 

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        field_name = "contributing_factors"
        plt.savefig(f'./graphs/{field_name}.png')
        plt.close()

    def contributing_factor_ai(self):
        x_2 = self.cause_data_lst
        (lst_contr_fact, lst_data, max_label, max_label_val) = self.group_up_str_fields(x_2, None)

        plt.bar(lst_contr_fact, lst_data)
        plt.ylim([0, max_label_val * 1.2])
        plt.xlabel("Cause of Accident")
        plt.ylabel("Number Reported")
        plt.title("Frequency of Different Causes of Accident")

        figure = plt.gcf()
        figure.set_size_inches(8, 6)
        field_name = "cause_of_accident"
        plt.savefig(f'./graphs/{field_name}.png')
        plt.close()

    '''
    TODO KMEANS WITH CAUSE OF ACCIDENT
    '''

    '''
    TODO KMEANS WITH VEHICLE INFORMATION
    '''

    def k_means_groups(self, lst_facts):
        burroughs_lst = ["BROOKLYN", "QUEENS", "MANHATTAN", "BRONX", "STATEN ISLAND"]
       
        a = lambda x: 0 if math.isnan(x) else x
        b = lambda x: burroughs_lst.index(x) + 1 if x in burroughs_lst else float(x)
        c = lambda x: a(x) if type(x) != str else b(x)
        d = lambda x: c(x) if x else 0 
        lst_dumb = []
        for val in np.array(self.df[lst_facts].values):
            lst_dumb += [np.array([d(val_2) for val_2 in val])]
        X = np.array(lst_dumb)
        print(X.shape)
        numCluster = 30
        kmeans = KMeans(n_clusters = numCluster).fit(X)

        print(f"Factors:    {lst_facts}")
        
        lstCount = [(0, i) for i in range(numCluster)]
        for label in kmeans.labels_:
            tV = lstCount[label]
            lstCount[label] = (tV[0] + 1, tV[1])
        lstCount = sorted(lstCount)[::-1]
        print("Centers:")
        for i in range(len(lstCount)):
            groupNum = lstCount[i][1]
            numCount = lstCount[i][0]
            print(f"{groupNum},   {numCount},    {[round(val,2) for val in kmeans.cluster_centers_[groupNum]]}")


    def all_kmeans_analysis(self):
        lst_1 = ["CRASH TIME","BOROUGH", "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED"]
        print("Deaths and Injuries")
        self.k_means_groups(lst_1)

    def introductory_analysis(self):
        num_entries = self.df.shape[0].compute()
        print(f"Num motor vehicle accident entries is {num_entries}")
        
        #self.day_of_week_analysis()
        #self.yearly_analysis()
        #self.borough_analysis()
        #self.injuries_analysis()
        #self.deaths_analysis()
        self.contributing_factor()
        #self.crashtime_analysis()
        self.contributing_factor_ai()
        self.all_kmeans_analysis()



if __name__ == "__main__":
    file_handler = File_Handler()
    file_handler.introductory_analysis()