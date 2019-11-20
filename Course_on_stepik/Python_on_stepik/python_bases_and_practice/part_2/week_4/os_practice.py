import os
import os.path

print(os.getcwd())
print(os.listdir())

print(os.path.exists('files.py'))
print(os.path.exists('random.py'))

print(os.path.isfile('files.py'))
print(os.path.isdir('files.py'))

print(os.path.abspath('files.py'))

os.chdir('test_dir')
print(os.getcwd())