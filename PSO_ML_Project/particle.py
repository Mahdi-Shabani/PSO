import numpy as np

class Particle:
    def __init__(self, dim):
        """
        ایجاد یک ذره جدید
        dim : تعداد ویژگی‌ها (این دیتاست ۴ ویژگی داره)
        """
        self.position = np.random.rand(dim)       
        self.velocity = np.random.uniform(-1, 1, dim)  
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def __str__(self):
        return f"Pos: {self.position}, vel: {self.velocity}, pBest: {self.best_position}"