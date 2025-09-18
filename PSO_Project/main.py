from pso import PSO

if __name__ == "__main__":
    
    num_particles = 30       
    dim = 2                  
    bounds = (-10, 10)       
    max_iter = 100           

    optimizer = PSO(num_particles, dim, bounds, max_iter)
    best_position, best_value = optimizer.optimize()

    print("\n نتیجه نهایی:")
    print("بهترین موقعیت پیدا شده =", best_position)
    print("مقدار تابع در این نقطه =", best_value)