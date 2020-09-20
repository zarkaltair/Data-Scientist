import os
import time
import tempfile


class File:
    def __init__(self, path_to_file):
        self.path_to_file = os.path.join(tempfile.gettempdir(), path_to_file)
        self.lines_counters = 0

        if not os.path.exists(self.path_to_file):
            with open(self.path_to_file, 'w') as f:
                f.write('')

    def read(self):
        with open(self.path_to_file, 'r') as f:
            text = f.read()
        return text

    def write(self, text):
        with open(self.path_to_file, 'w') as f:
            f.write(text)

    def __str__(self):
        return self.path_to_file

    def __add__(self, obj):
        with open(self.path_to_file, 'r') as f_1:
            text_1 = f_1.read()
        with open(obj.path_to_file, 'r') as f_2:
            text_2 = f_2.read()
        s = text_1 + text_2
        new_file_obj = File(str(time.time()))
        new_file_obj.write(s)
        return new_file_obj

    def __iter__(self):
        return self

    def __next__(self):
        with open(self.path_to_file, 'r') as f:
            contents = f.readlines()
        if self.lines_counters < len(contents):
            res = contents[self.lines_counters]
            self.lines_counters += 1
        else:
            self.lines_counters = 0
            raise StopIteration
        return res


path_to_file = 'some_filename'
print(os.path.exists(path_to_file))
# False
file_obj = File(path_to_file)
print(os.path.exists(path_to_file))
# True
print(file_obj.read())
''
print(file_obj.write('some text'))
# 9
print(file_obj.read())
# 'some text'
print(file_obj.write('other text'))
# 10
print(file_obj.read())
# 'other text'
file_obj_1 = File(path_to_file + '_1')
file_obj_2 = File(path_to_file + '_2')
print(file_obj_1.write('line 1\n'))
# 7
print(file_obj_2.write('line 2\n'))
# 7
new_file_obj = file_obj_1 + file_obj_2
print(isinstance(new_file_obj, File))
# True
print(new_file_obj)
# C:...
for line in new_file_obj:
    print(ascii(line))
# 'line 1\n'
# 'line 2\n'
