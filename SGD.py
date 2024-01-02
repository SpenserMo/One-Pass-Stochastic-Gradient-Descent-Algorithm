import numpy as np
import matplotlib.pyplot as plt
def generate_and_normalize_vectors_with_labels(N, sigma):
	# Define mean vectors
	mean_vector_pos = np.array([1/4, 1/4, 1/4, 1/4])
	mean_vector_neg = np.array([-1/4, -1/4, -1/4, -1/4])
 
	# Create the covariance matrix as sigma squared times the identity matrix
	covariance_matrix = np.square(sigma) * np.identity(4)
 
	normalized_vectors = np.zeros((N, 4))
	labels = np.zeros(N)
 
	for i in range(N):
    	# Randomly choose between the positive and negative mean vector of possibility 1/2 and assign the label
    	if np.random.rand() > 0.5:
        	chosen_mean = mean_vector_pos
        	labels[i] = 1
    	else:
        	chosen_mean = mean_vector_neg
        	labels[i] = -1
 
    	# Generate the Gaussian vector
    	vector = np.random.multivariate_normal(chosen_mean, covariance_matrix)
 
    	# Normalize the vector if its norm is greater than 1
    	normalized_vectors[i] = vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 1 else vector
 
	# Combine the vectors and labels
	vectors_with_labels = [(vector, label) for vector, label in zip(normalized_vectors, labels)]
 
	return vectors_with_labels
 
 
 
 
 
def sgd(T, sigma):
  lip = np.sqrt(2)
  step_size = 2 / (lip * np.sqrt(T))
 
  # Initialize w1 = 0
  w = np.zeros(5)
  w_T = np.zeros((T, 5))
 
  training_set = generate_and_normalize_vectors_with_labels(T, sigma)
 
  for t in range(T):
	# Draw a new training example z
	z = training_set[t]
 
	# Compute Gt
	x = np.append(training_set[t][0], 1)
	w_dot_x = np.dot(w, x)
	neg_y = -1 * training_set[t][1]
	numerator = neg_y * x * np.exp(neg_y * w_dot_x)
	denominator = 1 + np.exp(neg_y * w_dot_x)
	gradient = numerator / denominator
 
	# Update wt
	w = w - step_size * gradient
	w = w / np.linalg.norm(w) if np.linalg.norm(w) > 1 else w
 
	# Insert w into w_T
	w_T[t] = w
 
  # Return averaged w over w_T
  return w_T.sum(axis = 0) / T
 
 
 
 
 
def calculate_loss_and_err(n, sigma, testing_set):
  logistic_loss_30 = np.zeros(30)
  classification_error_30 = np.zeros(30)
 
  # Run SGD for 30 iterations
  for i in range(30):
	log_loss = 0
	class_err = 0
	w = sgd(n, sigma)
	# Evaluate 400 testing data for each iteration
	for j in range(400):
  	test_data = testing_set[j]
  	test_feature_vec = np.append(test_data[0], 1)
  	test_label = test_data[1]
  	w_dot_x = np.dot(w, test_feature_vec)
 
  	log_loss += np.log(1 + np.exp(-1 * test_label * w_dot_x))
  	class_err += 1 if w_dot_x * test_label < 0 else 0
 
	# Averaging the loss/err to get the expected loss/err
	log_loss /= 400
	class_err /= 400
 
	logistic_loss_30[i] = log_loss
	classification_error_30[i] = class_err
  # Return loss/err for 30 iterations
  return [logistic_loss_30, classification_error_30]
 

 
# Generate testing data for sigma=0.2 and sigma=0.4
testing_set1 = generate_and_normalize_vectors_with_labels(400, 0.2)
testing_set2 = generate_and_normalize_vectors_with_labels(400, 0.4)
 
# list of different sample sizes
x = [50, 100, 500, 1000]
# excess risk for different sample sizes for 0.2 variance
y1 = []
y1_mean = []
y1_min = []
y1_std = []
# classification error for different sample sizes for 0.2 variance
y2 = []
y2_std = []
# excess risk for different sample sizes for 0.4 variance
y3 = []
y3_mean = []
y3_min = []
y3_std = []
# classification error for different sample sizes for 0.4 variance
y4 = []
y4_std = []
 
# Measure loss/err for n=50, sigma=0.2
res = calculate_loss_and_err(50, 0.2, testing_set1)
loss_mean = np.mean(res[0])
loss_std = np.std(res[0])
loss_min = np.min(res[0])
loss_excess_risk = loss_mean - loss_min
y1.append(loss_excess_risk)
y1_mean.append(loss_mean)
y1_min.append(loss_min)
y1_std.append(loss_std)
 
err_mean = np.mean(res[1])
err_std = np.std(res[1])
y2.append(err_mean)
y2_std.append(err_std)
 
# Measure loss/err for n=100, sigma=0.2
res = calculate_loss_and_err(100, 0.2, testing_set1)
loss_mean = np.mean(res[0])
loss_std = np.std(res[0])
loss_min = np.min(res[0])
loss_excess_risk = loss_mean - loss_min
y1.append(loss_excess_risk)
y1_mean.append(loss_mean)
y1_min.append(loss_min)
y1_std.append(loss_std)
 
err_mean = np.mean(res[1])
err_std = np.std(res[1])
y2.append(err_mean)
y2_std.append(err_std)
 
# Measure loss/err for n=500, sigma=0.2
res = calculate_loss_and_err(500, 0.2, testing_set1)
loss_mean = np.mean(res[0])
loss_std = np.std(res[0])
loss_min = np.min(res[0])
loss_excess_risk = loss_mean - loss_min
y1.append(loss_excess_risk)
y1_mean.append(loss_mean)
y1_min.append(loss_min)
y1_std.append(loss_std)
 
err_mean = np.mean(res[1])
err_std = np.std(res[1])
y2.append(err_mean)
y2_std.append(err_std)
 
# Measure loss/err for n=1000, sigma=0.2
res = calculate_loss_and_err(1000, 0.2, testing_set1)
loss_mean = np.mean(res[0])
loss_std = np.std(res[0])
loss_min = np.min(res[0])
loss_excess_risk = loss_mean - loss_min
y1.append(loss_excess_risk)
y1_mean.append(loss_mean)
y1_min.append(loss_min)
y1_std.append(loss_std)
 
err_mean = np.mean(res[1])
err_std = np.std(res[1])
y2.append(err_mean)
y2_std.append(err_std)
 
# Measure loss/err for n=50, sigma=0.4
res = calculate_loss_and_err(50, 0.4, testing_set2)
loss_mean = np.mean(res[0])
loss_std = np.std(res[0])
loss_min = np.min(res[0])
loss_excess_risk = loss_mean - loss_min
y3.append(loss_excess_risk)
y3_mean.append(loss_mean)
y3_min.append(loss_min)
y3_std.append(loss_std)
 
err_mean = np.mean(res[1])
err_std = np.std(res[1])
y4.append(err_mean)
y4_std.append(err_std)
 
# Measure loss/err for n=100, sigma=0.2
res = calculate_loss_and_err(100, 0.4, testing_set2)
loss_mean = np.mean(res[0])
loss_std = np.std(res[0])
loss_min = np.min(res[0])
loss_excess_risk = loss_mean - loss_min
y3.append(loss_excess_risk)
y3_mean.append(loss_mean)
y3_min.append(loss_min)
y3_std.append(loss_std)
 
err_mean = np.mean(res[1])
err_std = np.std(res[1])
y4.append(err_mean)
y4_std.append(err_std)
 
# Measure loss/err for n=500, sigma=0.2
res = calculate_loss_and_err(500, 0.4, testing_set2)
loss_mean = np.mean(res[0])
loss_std = np.std(res[0])
loss_min = np.min(res[0])
loss_excess_risk = loss_mean - loss_min
y3.append(loss_excess_risk)
y3_mean.append(loss_mean)
y3_min.append(loss_min)
y3_std.append(loss_std)
 
err_mean = np.mean(res[1])
err_std = np.std(res[1])
y4.append(err_mean)
y4_std.append(err_std)
 
# Measure loss/err for n=1000, sigma=0.2
res = calculate_loss_and_err(1000, 0.4, testing_set2)
loss_mean = np.mean(res[0])
loss_std = np.std(res[0])
loss_min = np.min(res[0])
loss_excess_risk = loss_mean - loss_min
y3.append(loss_excess_risk)
y3_mean.append(loss_mean)
y3_min.append(loss_min)
y3_std.append(loss_std)
 
err_mean = np.mean(res[1])
err_std = np.std(res[1])
y4.append(err_mean)
y4_std.append(err_std)
 
# Print results for the table
print(y1_mean)
print(y1_min)
print(y1)
print(y1_std)
print('\n')
print(y2)
print(y2_std)
print('\n')
print(y3)
print(y3_mean)
print(y3_min)
print(y3_std)
print('\n')
print(y4)
print(y4_std)
 
# Plot figures
plt.figure(1)
plt.plot(x, y2, label = "sig 0.2 err", linestyle = '-')
plt.plot(x, y4, label = "sig 0.4 err", linestyle = ':')
 
plt.errorbar(x, y2, yerr = y2_std, fmt = 'o')
plt.errorbar(x, y4, yerr = y4_std, fmt = 'o')
 
plt.xlabel('Number of Training Samples')
plt.ylabel('Classification Error')
plt.title('Results measured by Classification Error')
 
plt.legend()
plt.show()
 
plt.figure(2)
plt.plot(x, y1, label = "sig 0.2 loss", linestyle = '-')
plt.plot(x, y3, label = "sig 0.4 loss", linestyle = ':')
 
plt.errorbar(x, y1, yerr = y1_std, fmt = 'o')
plt.errorbar(x, y3, yerr = y3_std, fmt = 'o')
 
plt.xlabel('Number of Training Samples')
plt.ylabel('Excess Risk')
plt.title('Results measured by Excess Risk')
 
plt.legend()
plt.show()
 
plt.figure(3)
plt.plot(x, y1, label = "sig 0.2 loss", linestyle = '-')
plt.errorbar(x, y1, yerr = y1_std, fmt = 'o')
plt.xlabel('n')
plt.ylabel('Loss/Error')
plt.legend()
plt.show()
 
plt.figure(4)
plt.plot(x, y2, label = "sig 0.2 err", linestyle = '--')
plt.errorbar(x, y2, yerr = y2_std, fmt = 'o')
plt.xlabel('n')
plt.ylabel('Loss/Error')
plt.legend()
plt.show()
 
plt.figure(5)
plt.plot(x, y3, label = "sig 0.4 loss", linestyle = '-.')
plt.errorbar(x, y3, yerr = y3_std, fmt = 'o')
plt.xlabel('n')
plt.ylabel('Loss/Error')
plt.legend()
plt.show()
 
plt.figure(6)
plt.plot(x, y4, label = "sig 0.4 err", linestyle = ':')
plt.errorbar(x, y4, yerr = y4_std, fmt = 'o')
plt.xlabel('n')
plt.ylabel('Loss/Error')
plt.legend()
plt.show()
