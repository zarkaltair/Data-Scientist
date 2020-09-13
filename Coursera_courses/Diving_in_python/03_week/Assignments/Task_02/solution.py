import os
import csv


class CarBase:

    def __init__(self, brand, photo_file_name, carrying):
        self.brand = brand
        self.photo_file_name = photo_file_name
        self.carrying = float(carrying)

    def get_photo_file_ext(self):
        arr_ext = ['.png','.jpeg', '.jpg', '.gif']
        ext = os.path.splitext(self.photo_file_name)[1]
        if ext in arr_ext:
            return ext


class Car(CarBase):

    car_type = 'car'

    def __init__(self, brand, photo_file_name, carrying, passenger_seats_count):
        super().__init__(brand, photo_file_name, carrying)
        self.passenger_seats_count = int(passenger_seats_count)


class Truck(CarBase):

    car_type = 'truck'

    def __init__(self, brand, photo_file_name, carrying, body_whl):
        super().__init__(brand, photo_file_name, carrying)
        try:
            if len(body_whl.split('x')) != 3:
                self.body_length = 0.0
                self.body_width = 0.0
                self.body_height = 0.0
            elif float(body_whl.split('x')[0]) and float(body_whl.split('x')[1]) and float(body_whl.split('x')[2]):
                self.body_length = float(body_whl.split('x')[0])
                self.body_width = float(body_whl.split('x')[1])
                self.body_height = float(body_whl.split('x')[2])
            else:
                self.body_length = 0.0
                self.body_width = 0.0
                self.body_height = 0.0
        except:
            self.body_length = 0.0
            self.body_width = 0.0
            self.body_height = 0.0


    def get_body_volume(self):
        return self.body_length * self.body_width * self.body_height


class SpecMachine(CarBase):

    car_type = 'spec_machine'

    def __init__(self, brand, photo_file_name, carrying, extra):
        super().__init__(brand, photo_file_name, carrying)
        self.extra = extra


def get_car_list(csv_filename):
    car_list = []
    with open(csv_filename) as csv_fd:
        reader = csv.reader(csv_fd, delimiter=';')
        next(reader)  # skip head

        for row in reader:
            if len(row) == 7:
                car_type, brand, passenger_seats_count, photo_file_name, body_whl, carrying, extra = row
                if car_type == 'car' and brand != '' and (os.path.splitext(photo_file_name)[0] and os.path.splitext(photo_file_name)[1]) != '' and ('.' not in os.path.splitext(photo_file_name)[0]) and passenger_seats_count.isdigit() and carrying != '' and float(carrying):
                    car_list.append(Car(brand, photo_file_name, carrying, passenger_seats_count))
                    # print('success append car type')

                elif car_type == 'truck' and brand != '' and (os.path.splitext(photo_file_name)[0] and os.path.splitext(photo_file_name)[1]) != '' and ('.' not in os.path.splitext(photo_file_name)[0]) and carrying != '' and float(carrying):
                    car_list.append(Truck(brand, photo_file_name, carrying, body_whl))
                    # print('success append truck type')

                elif row[0] == 'spec_machine' and brand != '' and photo_file_name != '' and (os.path.splitext(photo_file_name)[0] and os.path.splitext(photo_file_name)[1]) != '' and ('.' not in os.path.splitext(photo_file_name)[0]) and carrying != '' and float(carrying) and extra != '' :
                    car_list.append(SpecMachine(brand, photo_file_name, carrying, extra))
                    # print('success append spec machine type')

    return car_list


# car = Car('Bugatti Veyron', 'bugatti.png', '0.312', '2')
# print(car.car_type, car.brand, car.photo_file_name, car.carrying, car.passenger_seats_count, sep='\n')

# truck = Truck('Nissan', 'nissan.jpeg', '1.5', '3.92x2.09x1.87')
# print(truck.car_type, truck.brand, truck.photo_file_name, truck.body_length, truck.body_width, truck.body_height, sep='\n')

# spec_machine = SpecMachine('Komatsu-D355', 'd355.jpg', '93', 'pipelayer specs')
# print(spec_machine.car_type, spec_machine.brand, spec_machine.carrying, spec_machine.photo_file_name, spec_machine.extra, sep='\n')

# spec_machine.get_photo_file_ext()

# cars = get_car_list('cars.csv')
# print(len(cars))

# for car in cars:
#     print(type(car))

# print(cars[0])
# print(cars[0].passenger_seats_count)
# print(cars[1].get_body_volume())
# print(cars)
