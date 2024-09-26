import numpy as np

# z-->x, z-->y, o-->x, o-->y, x-->y
# def generate_observational_data_icm(num_samples):
#     z = np.zeros(num_samples)
#     o = np.zeros(num_samples)
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         prob = np.random.random()
#         prob2 = np.random.random()
#         z[i] = np.random.choice([0, 1], size=1, p=[prob, 1-prob])
#         o[i] = np.random.choice([0, 1], size=1, p=[prob2, 1-prob2])
#         x[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 and o[i] == 1 else np.random.choice([0, 1], p=[0.9, 0.1])
#         y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 and o[i] == 1 else np.random.choice([0, 1], p=[0.9, 0.1]) if x[i] == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
#     return np.concatenate((z.reshape(num_samples,1),o.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
# def generate_interventional_x_data_icm(num_samples):
#     z = np.zeros(num_samples)
#     o = np.zeros(num_samples)
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         prob = np.random.random()
#         prob2 = np.random.random()
#         z[i] = np.random.choice([0, 1], size=1, p=[prob, 1-prob])
#         o[i] = np.random.choice([0, 1], size=1, p=[prob2, 1-prob2])
#         x[i] = np.random.choice([0, 1])
#         y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 and o[i] == 1 else np.random.choice([0, 1], p=[0.9, 0.1]) if x[i] == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
#     return np.concatenate((z.reshape(num_samples,1),o.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)

def generate_observational_data_icm(num_samples):
    z = np.zeros(num_samples)
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    for i in range(num_samples):
        prob = np.random.random()
        z[i] = np.random.choice([0, 1], size=1, p=[prob, 1-prob])
        x[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
        y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1]) if x[i] == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
    return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)

def generate_interventional_x_data_icm(num_samples):
    z = np.zeros(num_samples)
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    for i in range(num_samples):
        prob = 0.1#np.random.random()
        z[i] = np.random.choice([0, 1], size=1, p=[prob, 1-prob])
        x[i] = np.random.choice([0, 1], p=[prob, 1-prob])
        y[i] = np.random.choice([0, 1], p=[0.5, 0.5]) if z[i] == 0 else np.random.choice([0, 1], p=[0.7, 0.3]) if x[i] == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
    return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
