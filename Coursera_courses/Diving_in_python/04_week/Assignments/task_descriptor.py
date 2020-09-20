class Value:
    @staticmethod
    def _prepare_value(amount, commission):
        return amount * commission

    def __get__(self, obj, obj_type):
        return self.value

    def __set__(self, obj, value):
        if obj.commission > 1:
            self.value = 0
        else:
            self.value = int(value - self._prepare_value(value, obj.commission))


class Account:
    amount = Value()
    
    def __init__(self, commission):
        self.commission = commission


# new_account = Account(0.1)
# new_account.amount = 100

# print(new_account.amount)
# # 90

# new_account.commission = 0
# new_account.amount = 50

# print(new_account.amount)
# # 50

# new_account.commission = 1
# new_account.amount = 50

# print(new_account.amount)
# # 0

# new_account.commission = 1.1
# new_account.amount = 50

# print(new_account.amount)
# # 0
