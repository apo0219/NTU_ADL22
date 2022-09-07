import json
import csv
import sys
# print( sys.argv[0], sys.argv[1] )
f_valid = open( sys.argv[1], "r", encoding="utf-8" )
f_context = open( sys.argv[2], "r", encoding="utf-8" )
data_context = json.load( f_context )
data_train = json.load( f_valid )
outdata = []
for i in data_train:
    id = []
    outdata.append( [ i["id"], i["question"], data_context[i["paragraphs"][0]], data_context[i["paragraphs"][1]], data_context[i["paragraphs"][2]], data_context[i["paragraphs"][3]], "0" ] )
    # print( [ i["question"], i["paragraphs"][0], i["paragraphs"][1], i["paragraphs"][2], i["paragraphs"][3], i["paragraphs"].index( i["relevant"] ) ] )

with open("mc_test.csv", "w", newline="", encoding="utf-8" ) as f_csv:
    writer = csv.writer( f_csv )
    writer.writerow( [ "id", "sent1", "ending0", "ending1", "ending2", "ending3", "label" ] )
    for i in outdata:
        writer.writerow( i )
