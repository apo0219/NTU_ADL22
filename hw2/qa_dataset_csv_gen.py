from dataclasses import dataclass
import json
import csv
import sys
f_valid = open( sys.argv[1], "r", encoding="utf-8" )
f_context = open( sys.argv[2], "r", encoding="utf-8" )
data_context = json.load( f_context )
data_train = json.load( f_valid )
outdata = []
for i in data_train:
    # outdata.append( [ i["id"], i["question"], data_context[i["paragraphs"][0]], data_context[i["paragraphs"][1]], data_context[i["paragraphs"][2]], data_context[i["paragraphs"][3]], i["paragraphs"].index( i["relevant"] ), data_context[i["relevant"]], i["answer"]["start"], i["answer"]["text"] ] )
    # print( i["id"], i["question"], data_context[i["paragraphs"][0]], data_context[i["paragraphs"][1]], data_context[i["paragraphs"][2]], data_context[i["paragraphs"][3]], i["paragraphs"].index( i["relevant"] ), data_context[i["relevant"]], i["answer"]["start"], i["answer"]["text"], sep="\n" )
    outdata.append( [ i["id"], i["question"], data_context[i["relevant"]], i["answer"]["start"], i["answer"]["text"] ] )
    print( i["id"], i["question"], data_context[i["relevant"]], i["answer"]["start"], i["answer"]["text"], sep="\n" )


with open( argv[3], "w", newline="", encoding="utf-8" ) as f_csv:
    writer = csv.writer( f_csv )
    writer.writerow( [ "id", "question", "context", "answer_start", "answer_text" ] )
    for i in outdata:
        writer.writerow( i )    
