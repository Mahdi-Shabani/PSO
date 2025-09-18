from pso import PSO
import numpy as np

if __name__ == "__main__":
    num_particles = 20   
    dim = 4              
    max_iter = 50        

    optimizer = PSO(num_particles, dim, max_iter)
    best_position, best_value = optimizer.optimize()

    print("\n نتیجه نهایی:")
    print("بهترین subset ویژگی‌ها =", best_position >= 0.5)
    print("هزینه (1 - دقت مدل) =", best_value)
    print("دقت مدل =", 1 - best_value)