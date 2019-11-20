import os
import os.path
import shutil

shutil.copy('test1.txt', 'test2.txt')

shutil.copytree('test_dir', 'test_dir/tests')

for current_dir, dirs, files in os.walk('.'):
    print(current_dir, dirs, files)