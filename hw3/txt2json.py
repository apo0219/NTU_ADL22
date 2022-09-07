import json
import argparse
from pathlib import Path
def main( args ):
    print( "args.dir : ", args.dir )
    with open('input.json', 'r', encoding="utf-8" ) as f:
        data = [json.loads(line) for line in f]

    predict_file = Path( args.dir ) / 'generated_predictions.txt'
    with open( predict_file, 'r', encoding='utf-8' ) as f:
        titles = []
        line = f.readline()
        while ( line ):
            titles.append( line )
            line = f.readline()
    with open( Path( args.dir ) / 'predict.json', 'w') as f:
        count = 0
        for i in range( len( data ) ) :
            count+=1
            if ( count % 1000 == 0 ): print( count )
            out = {
                'title' : titles[i],
                'id' : data[i]['id'],
            }
            json.dump( out, f, ensure_ascii=False )
            f.write('\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '-d', '--dir' )
    args = parser.parse_args()
    main( args )