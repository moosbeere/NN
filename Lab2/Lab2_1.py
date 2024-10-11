import numpy as np

arr1 = np.array([1,2,3,4])
print(arr1)

arr2 = np.array([[1,2,3,4], [5,6,7,8]])
print(arr2)
print(arr2.shape)

zeros = np.zeros((3,3))
print(zeros)

ones = np.ones((3,3))
print(ones)

print(np.ones_like(arr2))

arr5 = np.random.rand(2,3)
print(arr5)

arr1  = np.array([1,2,3,4])
arr2  = np.array([3,4,5,6])

result = arr1 - arr2
result = arr1 + arr2
result = arr1 * arr2
result = arr1 / arr2
print(result)

print(arr1[0])
print(arr2[1])
slice = arr1[2:3]
print(slice)

mask = arr1 > 2
print(arr1[mask])

print(arr2[arr2 < 5])

mean = np.mean(arr1)
print(mean)

std = np.std(arr2)
print(std)

max_value = np.max(arr2)
min_value = np.min(arr2)

print(min_value)
print(max_value)

arr2 = np.array([[1,2,3], [5,6,7]])
arr3 = np.array([[5,6], [7,8], [9,10]])
print(arr2.dot(arr3))

print(arr2.T)



