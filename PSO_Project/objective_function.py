import numpy as np

def sphere(position):
    """
    تابع هدف ساده: Sphere Function
    f(x, y, ...) = sum(x_i^2)
    ورودی:
        position : لیست یا numpy array از مختصات ذره (x, y, ...)
    خروجی:
        مقدار تابع در آن نقطه (scalar)
    """
    return np.sum(np.square(position))


# ============================
# test
# ============================
# if __name__ == "__main__":
#    pos1 = np.array([0, 0])     # انتظار: 0^2 + 0^2 = 0
#    pos2 = np.array([2, 3])     # انتظار: 2^2 + 3^2 = 13

#    print("sphere([0,0]) =", sphere(pos1))
#    print("sphere([2,3]) =", sphere(pos2))