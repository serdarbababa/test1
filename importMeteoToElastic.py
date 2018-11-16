import numpy as np
import pandas as pd
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch import helpers

station = "15462"
es = Elasticsearch([{'host': '192.168.229.131', 'port': 9200}])

df = pd.read_csv('/home/u1/meteo'+station+'.csv')
print df.head()

#es.create_index("meteoData")
df = df.dropna(subset=["Time (UTC)","Barometric Pressure (mbar)","Temperature (degrees F)","Dewpoint (degrees F)","Relative Humidity (%)","Wind speed (mph)","Wind direction (degrees)"])

for i in range(len(df)):
        #print (str(df["date"][i]), df["Open"][i], df["High"][i], df["Low"][i], df["Close"][i], df["Volume_(BTC)"][i], df["Volume_(Currenc$
        doc={
                "@timestamp":  datetime.strptime(df["Time (UTC)"][i], '%Y-%m-%d %H:%M:%S'),
                "Pressure": df["Barometric Pressure (mbar)"][i],
                "Temperature": df["Temperature (degrees F)"][i],
                "Humidity": df["Relative Humidity (%)"][i],
                "WindSpeed": df["Wind speed (mph)"][i],
                "WindDirection": df["Wind direction (degrees)"][i],
                "Station": station	

                }
	print doc	
        #index = '{"index":{"_index":"btc",     "_type":"log"}}'
        res = es.index(index="meteo-"+station, doc_type='doc', id=i, body=doc)



