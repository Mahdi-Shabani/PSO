import numpy as np
from particle import Particle
from objective_function import sphere


class PSO:
    def __init__(self, num_particles, dim, bounds, max_iter, w=0.5, c1=1.5, c2=1.5):
        """
        الگوریتم PSO
        پارامترها:
            num_particles : تعداد ذره‌ها
            dim           : تعداد ابعاد (مثلا 2 → [x,y])
            bounds        : بازه‌ی جستجو (min, max)
            max_iter      : تعداد تکرار (iteration)
            w             : ضریب اینرسی (اثر حرکت قبلی)
            c1            : ضریب یادگیری فردی
            c2            : ضریب یادگیری جمعی
        """
        self.num_particles = num_particles
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = [Particle(dim, bounds) for _ in range(num_particles)]

        self.global_best_position = np.random.uniform(bounds[0], bounds[1], dim)
        self.global_best_value = float('inf')


    def optimize(self):
        """ اجرای الگوریتم PSO روی تابع Sphere """

        for iteration in range(self.max_iter):
            for particle in self.particles:
                fitness = sphere(particle.position)

                if fitness < particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = np.copy(particle.position)

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

            for particle in self.particles:
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)

                particle.velocity = self.w * particle.velocity + cognitive + social

                particle.position = particle.position + particle.velocity

                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: gBest = {self.global_best_value}")

        return self.global_best_position, self.global_best_value