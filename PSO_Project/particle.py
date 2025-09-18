import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        """
        ایجاد یک ذره جدید
        پارامترها:
            dim    : تعداد ابعاد (مثلا 2 به معنی [x,y])
            bounds : تاپل (min, max) برای مقداردهی اولیه
        """
        self.position = np.random.uniform(bounds[0], bounds[1], dim)  
        self.velocity = np.random.uniform(-1, 1, dim)                 
        self.best_position = np.copy(self.position)                   
        self.best_value = float('inf')                                

    def __str__(self):
        return f"Pos: {self.position}, Vel: {self.velocity}, pBest: {self.best_position}"