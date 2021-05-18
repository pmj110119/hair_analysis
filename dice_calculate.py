

def run(target):
    left = 1
    right = 2
    s = 3       # 初始 [1,2]=3
    result = []
    while left < right:
        if s >= target: # 1.上一次sum等于target，left向右+1接着找    2.上一次sum大于target，left向右+1来减小sum，接着找
            s -= left
            left += 1
        else:
            right += 1
            s += right
        if s == target: # 匹配成功
            result.append(list(range(left, right + 1)))
    return result

print(run(100))
