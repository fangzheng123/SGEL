# encoding: utf-8


result = [0.85, 0.83, 0.88, 0.89, 0.80, 0.68, 0.68, 0.71]

num = [4791, 4485, 727, 257, 747, 144, 523, 634]

sum_result = 0
for a, b in zip(result, num):
    sum_result += a * b

avg = sum_result / sum(num)

print(avg)