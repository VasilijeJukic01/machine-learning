from PIL import Image
import numpy as np

# Init values
iterations = 1
img_dim = 2500
beta = 200

# Import images and damaged images
patterns = []
dmg_patterns = []

for i in range(1, 4):
    pattern = Image.open(f"pattern{i}.png")
    pattern = pattern.convert("L")
    pattern_arr = np.array(pattern)

    pattern_arr = pattern_arr.flatten()

    patterns.append(np.where(pattern_arr == 255, -1, 1))

for i in range(1, 4):
    dmg_pattern = Image.open(f"pattern{i}Dmg.png")
    dmg_pattern = dmg_pattern.convert("L")
    dmg_pattern_arr = np.array(dmg_pattern)

    dmg_pattern_arr = dmg_pattern_arr.flatten()
    dmg_patterns.append(np.where(dmg_pattern_arr == 255, -1, 1))

# Output saver
def save_image(x, filename):
    image_data = np.where(x == 1, 0, 255).astype(np.uint8)
    image_data = np.reshape(image_data, (50, 50))
    image = Image.fromarray(image_data, 'L')
    image.save(filename)

# Calculating weights
z = np.vstack(patterns)

w = np.zeros((img_dim, img_dim))
for k in range(len(z)):
    w += np.outer(z[k], z[k])
w -= len(z) * np.eye(img_dim)

# for k in range(len(z)):
#     for i in (range(len(z[k]))):
#         for j in (range(i)):
#             w[i][j] += z[k][i] * z[k][j]
#             w[j][i] += z[k][i] * z[k][j]

w /= img_dim
print(w)

# Activation function
def tanh(x):
    return (1 - np.exp(-beta * x)) / (1 + np.exp(-beta * x))

activation = np.vectorize(tanh)

# Output
def calculate_output(x, w):
    u = w.dot(x)
    v = activation(u)
    return v

# Restore function
def restore(damaged, index):
    damaged = damaged[np.newaxis].transpose()

    for i in range(iterations):
        prev = np.ones((img_dim, 1))
        phase = 1

        while np.sum(np.abs(prev - damaged)) > 1e-8:
            prev = damaged
            damaged = calculate_output(damaged, w)

            damaged_matrix = np.reshape(damaged, (50, 50))
            save_image(damaged_matrix, f"output/phases/restoration{index + 1}_phase{phase}.png")

            phase += 1

    damaged_matrix = np.reshape(damaged, (50, 50))
    save_image(damaged_matrix, f"output/restoration{index+1}.png")

# Driver
for i in range(len(dmg_patterns)):
    restore(dmg_patterns[i], i)
