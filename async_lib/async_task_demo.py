#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
async_task_demo_v14.py
演示异步任务流式(stream)与非流式(non-stream)返回
特点:
  - stream=True: yield 每条结果
  - stream=False: yield 每个生成器的全部步骤
调用方式:
  python async_task_demo_v14.py stream
  python async_task_demo_v14.py nonstream
"""

import asyncio
import time
from typing import AsyncGenerator, List, Dict, Union

# ---------------------------
# 模拟任务状态
# ---------------------------
class TaskState:
    def __init__(self):
        self.out_list: List[Dict] = []
        self.finished = False
        self.event = asyncio.Event()

# ---------------------------
# 异步任务生成器
# ---------------------------
class AsyncTask:
    def __init__(self, name: str, delay: float, index: int):
        self.name = name
        self.delay = delay
        self.index = index
        self.current_step = 0
        self.total_steps = 3
        self.state = TaskState()

    async def run(self):
        while self.current_step < self.total_steps:
            await asyncio.sleep(self.delay)
            self.state.out_list.append({
                "index": self.index,
                "text": f"{self.name} step {self.current_step} @ {time.perf_counter():.2f}"
            })
            self.state.event.set()
            self.state.event.clear()
            self.current_step += 1

        # done
        await asyncio.sleep(self.delay)
        self.state.out_list.append({
            "index": self.index,
            "text": f"{self.name} done @ {time.perf_counter():.2f}"
        })
        self.state.finished = True
        self.state.event.set()

    async def wait_one_response(self, stream: bool = True) -> AsyncGenerator[Union[Dict, List[Dict]], None]:
        if stream:
            # 流式模式: 每条 yield
            while True:
                await self.state.event.wait()
                out = self.state.out_list.pop(0)
                yield out
                if self.state.finished and not self.state.out_list:
                    break
                self.state.event.clear()
        else:
            # 非流式模式: 等待任务完成，再一次性返回平铺列表
            while not self.state.finished:
                await self.state.event.wait()
                self.state.event.clear()
            # 注意这里 yield **平铺列表**
            yield list(self.state.out_list)

# ---------------------------
# 核心函数 (handle_tasks 不变)
# ---------------------------
async def handle_tasks(stream: bool = True) -> AsyncGenerator[Union[Dict, List[Dict]], None]:
    specs = [("A", 1), ("B", 1), ("C", 1)]
    tasks = [AsyncTask(name, delay, idx) for idx, (name, delay) in enumerate(specs)]

    for t in tasks:
        asyncio.create_task(t.run())

    generators = [t.wait_one_response(stream=stream) for t in tasks]

    if stream:
        task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
        while task_map:
            done, _ = await asyncio.wait(task_map.keys(), return_when=asyncio.FIRST_COMPLETED)
            for tsk in done:
                gen = task_map.pop(tsk)
                try:
                    result = tsk.result()
                    yield result
                    task_map[asyncio.create_task(gen.__anext__())] = gen
                except StopAsyncIteration:
                    pass
    else:
        # 非流式模式，保持 handle_tasks 逻辑不变
        outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
        # 平铺所有生成器的结果
        flat_results = [item for sublist in outputs for item in sublist]
        yield flat_results

# ---------------------------
# 演示调用
# ---------------------------
async def run_demo(stream: bool = True):
    start = time.perf_counter()
    async for val in handle_tasks(stream):
        if isinstance(val, list):
            print("[Batch] Generated text:")
            for v in val:
                print(f"index={v['index']}: {v['text']}")
        else:
            print(f"[Stream] index={val['index']}: {val['text']}")
    end = time.perf_counter()
    print(f"Time elapsed: {end - start:.2f}s")

# ---------------------------
# 入口
# ---------------------------
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "stream"
    asyncio.run(run_demo(stream=(mode == "stream")))
