import csv
import pandas as pd
from mlp_wakati import Preprocessing
pp = Preprocessing()

class Connector:

    def return_ids(self,filename):
        text = self.csv_opener(filename)
        return pp.text_to_id(text[0]),text[1]

    #データの読み込み
    def csv_opener(self,csv_name):
        text = []
        with open(csv_name, 'r', encoding="utf_8_sig") as f:
            reader = csv.reader(f)
            for row in reader:
                text.append(row)
            return pd.DataFrame(text).T.values.tolist()

    