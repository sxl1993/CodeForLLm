from typing import List
"""
704. 二分查找
"""
def search_v1(nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            middle = left + (right - left) // 2
            if nums[middle] > target:
                right = middle - 1
            elif nums[middle] < target:
                left = middle + 1
            else:
                return middle
        return -1
    

def search_v2(nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)
        while left < right:
            middle = left + (right - left) // 2
            if nums[middle] > target:
                right = middle
            elif nums[middle] < target:
                left = middle + 1
            else:
                return middle
        return -1  

        
if __name__ == '__main__':
    nums = [5]
    target = 5
    print(search_v1(nums, target))
    print(search_v2(nums, target))