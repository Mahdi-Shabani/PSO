import numpy as np
from particle import Particle
from objective_function import feature_selection_objective

class PSO:
    def __init__(self, num_particles, dim, max_iter, w=0.5, c1=1.5, c2=1.5):
        self.num_particles = num_particles
        self.dim = dim
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles = [Particle(dim) for _ in range(num_particles)]
        self.global_best_position = np.random.rand(dim)
        self.global_best_value = float('inf')

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.particles:
                fitness = feature_selection_objective(particle.position)

                if fitness < particle.best_value:
                    particle.best_value = fitness
                    particle.best_position = np.copy(particle.position)

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = np.copy(particle.position)

            for particle in self.particles:
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)

                cognitive = self.c1 * r1 * (particle.best_position - particle.position)
                social = self.c2 * r2 * (self.global_best_position - particle.position)

                particle.velocity = self.w * particle.velocity + cognitive + social
                particle.position = particle.position + particle.velocity

                particle.position = np.clip(particle.position, 0, 1)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}: gBest = {self.global_best_value}")

        return self.global_best_position, self.global_best_value