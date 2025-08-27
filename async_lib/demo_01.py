import asyncio

import time
from datetime import datetime

def task_func():
    print("开始执行异步函数", datetime.now())
    # await asyncio.sleep(4)
    print("执行异步函数结束", datetime.now())

async def delay(seconds):
    print(f"开始休眠 {seconds} 秒", datetime.now())
    task_func()
    # t1 = asyncio.create_task(task_func())
    await asyncio.sleep(seconds)
    # await t1
    print(f"休眠完成", datetime.now())
    return seconds

async def main():
    # 将 delay(3) 包装成任务，注：包装完之后直接就丢到事件循环里面运行了
    # 因此这里会立即返回，而返回值是一个 asyncio.Task 对象
    sleep_for_three = asyncio.create_task(delay(3))
    print("sleep_for_three:", sleep_for_three.__class__)
    # 至于协程究竟有没有运行完毕，我们可以通过 Task 对象来查看
    # 当协程运行完毕或者报错，都看做是运行完毕了，那么调用 Task 对象的 done 方法会返回 True
    # 否则返回 False，由于代码是立即执行，还没有到 3 秒钟，因此打印结果为 False
    print("协程(任务)是否执行完毕:", sleep_for_three.done())
    # 这里则保证必须等到 Task 对象里面的协程运行完毕后，才能往下执行
    result = await sleep_for_three
    print("协程(任务)是否执行完毕:", sleep_for_three.done(), datetime.now())
    print("返回值:", result)

asyncio.run(main())
"""
sleep_for_three: <class '_asyncio.Task'>
协程(任务)是否执行完毕: False
开始休眠 3 秒
休眠完成
协程(任务)是否执行完毕: True
返回值: 3
"""
