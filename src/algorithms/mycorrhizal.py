import matplotlib.pyplot as plt


class DMOA:
    def __init__(self, a, b, d, g, c, e, f, x0, y0, epochs, iterations):
        self.a = a
        self.b = b
        self.d = d
        self.g = g
        self.c = c
        self.e = e
        self.f = f
        self.x0 = x0
        self.y0 = y0
        self.epochs = epochs
        self.iterations = iterations

    def predator_prey_model(self, x, y):
        x_next = self.a * x * (1 - x) - self.b * x * y
        y_next = self.d * x * y - self.g * y
        return x_next, y_next

    def cooperative_model(self, x, y):
        x_next = (self.a * x - self.b * x * y) / (1 - self.g * x)
        y_next = (self.d * y + self.c * y * y) / (1 + y)
        return x_next, y_next

    def competitive_model(self, x, y):
        x_next = (self.a * x - self.b * x * y) / (1 + self.g * y)
        y_next = (self.d * y - self.c * y * y) / (1 + y)
        return x_next, y_next

    def run_simulation(self, model_type):
        x_vals = []
        y_vals = []
        x = self.x0
        y = self.y0

        for _ in range(self.epochs):
            for _ in range(self.iterations):
                if model_type == "predator_prey":
                    x, y = self.predator_prey_model(x, y)
                elif model_type == "cooperative":
                    x, y = self.cooperative_model(x, y)
                elif model_type == "competitive":
                    x, y = self.competitive_model(x, y)

                x_vals.append(x)
                y_vals.append(y)

        return x_vals, y_vals


# Parameters
a, b, d, g, c, e, f = 0.01, 0.02, 0.06, 0.09, 1.70, 0.09, 0.09
x0, y0 = 0.0002, 0.0006
epochs, iterations = 30, 100

# Initialize DMOA
dmoa = DMOA(a, b, d, g, c, e, f, x0, y0, epochs, iterations)

# Run simulations
x_predator_prey, y_predator_prey = dmoa.run_simulation("predator_prey")
x_cooperative, y_cooperative = dmoa.run_simulation("cooperative")
x_competitive, y_competitive = dmoa.run_simulation("competitive")

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x_predator_prey, label="Plants (Predator-Prey)")
plt.plot(y_predator_prey, label="Fungi (Predator-Prey)")
plt.title("Predator-Prey Model")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x_cooperative, label="Plants (Cooperative)")
plt.plot(y_cooperative, label="Fungi (Cooperative)")
plt.title("Cooperative Model")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x_competitive, label="Plants (Competitive)")
plt.plot(y_competitive, label="Fungi (Competitive)")
plt.title("Competitive Model")
plt.legend()

plt.tight_layout()
plt.show()
