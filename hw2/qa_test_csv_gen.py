import json
import csv
import sys
f_valid = open( sys.argv[1], "r", encoding="utf-8" )
f_context = open( sys.argv[2], "r", encoding="utf-8" )
data_context = json.load( f_context )
data_train = json.load( f_valid )
outdata = []
with open( "./mc_out/out.txt", 'r' ) as f:
    pre_id = f.readline()
lst = pre_id.split(',')
del lst[-1]
lst =list(map(int, lst))
count = 0
for i in data_train: 
    outdata.append( [ i["id"], i["question"], data_context[i["paragraphs"][lst[count]]], "0", " " ] )
    count += 1

with open("./qa_test.csv", "w", newline="", encoding="utf-8" ) as f_csv:
    writer = csv.writer( f_csv )
    writer.writerow( [ "id", "question", "context", "answer_start", "answer_text" ] )
    for i in outdata:
        writer.writerow( i )    
