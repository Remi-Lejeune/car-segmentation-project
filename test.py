from image_dataset import *

files_2 = get_files()
files_a = get_test_files()

#tensor of dimension 9x1x1 with values of 1
tensor = torch.ones(9, 1, 1)
weights = torch.tensor([1,2,3,4,5,6,7,8,9])

for i in range(len(weights)):
    tensor[i] = tensor[i] * weights[i]

print(tensor)
# print(files_2)
# print(len(files_2))
# print("\n")
# print(files_a)
# print(len(files_a))
# print("\n")
# print(get_clean_files())
# print(len(get_clean_files()))