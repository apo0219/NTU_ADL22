import json
import argparse
from pathlib import Path
def main( args ) :
    with open( args.filename, 'r', encoding="utf-8" ) as f:
        data = [json.loads(line) for line in f]
    with open( "input.json" , 'w') as f:
        for i in data:
            json.dump( i, f, ensure_ascii=False )
            f.write('\n')
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument( '--filename', type=Path, default='./predict.jsonl' )
    args = parser.parse_args()
    main( args )

