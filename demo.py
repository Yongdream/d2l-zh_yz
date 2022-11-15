import time
class A:
    def __del__(self):
        # 当对象被销毁时，会自动调用这个方法
        print('__del__ method is called')
a1 = A()
a2 = a1
a3 = a1
del a1
for x in range(2):
    print("waiting...")
    time.sleep(1)
del a2
print("del a2")
for x in range(2):
    print("waiting...")
    time.sleep(1)
del a3
# del x 并不直接调用 x.__del__().前者会将 x 的引用计数减一，而后者仅会在 x 的引用计数变为零时被调用。

