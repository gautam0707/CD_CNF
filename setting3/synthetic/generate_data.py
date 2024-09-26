import numpy as np

# Define parameters
num_samples = 1000

# x-->y
# def generate_observational_data_icm(num_samples):
#     z = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)

#     for i in range(num_samples):
#        prob = np.random.random()
#        prob1 = np.random.random()
#        prob2 = np.random.random()     
#        x[i] = np.random.choice([0, 1], p=[prob, 1-prob])
#        y[i] = np.random.choice([0, 1], p=[prob1, 1-prob1]) if x[i] == 0 else np.random.choice([0, 1], p=[prob2, 1-prob2])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)

# z-->x, z-->y
# def generate_observational_data_icm(num_samples):

#     z = np.zeros(num_samples)
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         prob=np.random.random()
#         z[i] = np.random.choice([0, 1], size=1, p=[prob, 1-prob])
#         x[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
#         y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)


# z-->x, z-->y, x-->y
# def generate_observational_data_icm(num_samples):
#     z = z = np.zeros(num_samples)
#     x = np.zeros(num_samples)
#     y = np.zeros(num_samples)
#     for i in range(num_samples):
#         prob=np.random.random()
#         z[i] = np.random.choice([0, 1], size=1, p=[prob, 1-prob])
#         x[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1])
#         y[i] = np.random.choice([0, 1], p=[0.1, 0.9]) if z[i] == 0 else np.random.choice([0, 1], p=[0.9, 0.1]) if x[i] == 0 else np.random.choice([0, 1], p=[0.2, 0.8])
#     return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)

# z,x,y are independent
def generate_observational_data_icm(num_samples):
    z = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
    x = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
    y = np.random.choice([0, 1], size=num_samples, p=[0.5, 0.5])
    return np.concatenate((z.reshape(num_samples,1),x.reshape(num_samples,1),y.reshape(num_samples,1)), axis=1)
