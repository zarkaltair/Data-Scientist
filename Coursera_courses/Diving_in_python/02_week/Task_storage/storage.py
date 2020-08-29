import os
import json
import argparse
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--key', help='key name')
parser.add_argument('--val', help='value')
args = parser.parse_args()

dct = {}
dct[args.key] = [args.val]

storage_path = os.path.join(tempfile.gettempdir(), 'storage.data')

if args.val:

    if os.path.exists(storage_path):
        with open(storage_path, 'r') as f:
            dd = json.loads(f.read())
        
        if args.key not in dd:
            dd[args.key] = [args.val]
        
        else:
            dd[args.key].append(args.val)

        with open(storage_path, 'w') as f:
            f.write(json.dumps(dd))
    
    else:
        with open(storage_path, 'w') as f:
            f.write(json.dumps(dct))

else:
    if os.path.exists(storage_path):
        with open(storage_path, 'r') as f:
            dict_from_storage = json.loads(f.read())
            # print(dict_from_storage)
            for key, val in dict_from_storage.items():
                if key == args.key:
                    s = ', '.join(val)
                    print(s)
    else:
        print(None)
