#!/bin/bash

# First argument -- dir with files
# Second argiment -- bachup dir for these files

return_code=0
for file in `ls $1`
do
  echo "Found $file"
  if `rm $2/${file} 2>>$2/err.txt`
  then
    echo "Backup removed!"
  else
    echo "Can't remove backup!"
  fi
done

exit $return_code

