import numpy as np

# x-->y
# def generate_observational_data(num_samples):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         x[i] = np.random.choice([0, 1], p=[0.5, 0.5])
#         y[i] = np.random.choice([0, 1], p=[0.5, 0.5]) if x[i] == 0 else np.random.choice([0, 1], p=[0.5, 0.5])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_data_x(num_samples, intervention):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         x[i] = intervention
#         y[i] = np.random.choice([0, 1], p=[0.5, 0.5]) if x[i] == 0 else np.random.choice([0, 1], p=[0.5, 0.5])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_data_y(num_samples, intervention):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         x[i] = np.random.choice([0, 1], p=[0.5, 0.5])
#         y[i] = intervention
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)

# z-->x, z-->y
# def generate_observational_data(num_samples):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         x[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
#         y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_data_x(num_samples, intervention):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         x[i] = intervention
#         y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_data_y(num_samples, intervention):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         x[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
#         y[i] = intervention
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)


# z-->x, z-->y, x-->y
def generate_observational_data(num_samples):
    z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    for i in range(num_samples):
        x[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
        y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1]) if x[i] == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
    return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_data_x(num_samples, intervention):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         x[i] = np.random.choice([intervention, intervention], p=[0.5, 0.5])
#         y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1]) if x[i] == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_data_y(num_samples, intervention):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         x[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
#         y[i] = np.random.choice([intervention, intervention], p=[0.5, 0.5])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)

# z,x,y are independent
# def generate_observational_data(num_samples):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     y = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_data_x(num_samples, intervention):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.random.choice([intervention, intervention], size=num_samples, p=[0.2, 0.8])
#     y = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_data_y(num_samples, intervention):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     y = np.random.choice([intervention, intervention], size=num_samples, p=[0.2, 0.8])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
