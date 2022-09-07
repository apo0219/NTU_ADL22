import json
import csv
import sys
f_json = open( "./qa_out/predict_predictions.json", "r", encoding="utf-8" )
data_json = json.load( f_json )
# print(data_json)
# print(type(data_json))
outdata = []
for i in data_json:
    # print(i, data_json[i], sep=" ")
    outdata.append( [i, data_json[i]] )

with open( sys.argv[1] , "w", newline="", encoding="utf-8" ) as f_csv:
    writer = csv.writer( f_csv )
    writer.writerow( ["id", "answer"] )
    for i in outdata:
        writer.writerow( i )
