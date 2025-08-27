#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
async_task_demo_queue_unified.py
演示异步任务流式(stream)与批量(non-stream)返回
- 统一逻辑，非流式只是内部收集列表
- 流式模式可实时迭代
- 非流式模式等待所有任务完成再返回列表
"""

import asyncio
import time
from typing import AsyncGenerator, List, Dict, Union

# ---------------------------
# 异步任务
# ---------------------------
async def async_task(name: str, delay: float, queue: asyncio.Queue, index: int):
    for i in range(3):
        await asyncio.sleep(delay)
        await queue.put({"index": index, "text": f"{name} step {i} @ {time.perf_counter():.2f}"})
    await asyncio.sleep(delay)
    await queue.put({"index": index, "text": f"{name} done @ {time.perf_counter():.2f}", "done": True})

# ---------------------------
# 核心生成器
# ---------------------------
async def _task_generator(stream: bool = True) -> AsyncGenerator[Union[Dict, List[Dict]], None]:
    """
    流式和非流式统一逻辑
    """
    task_names = ["A", "B", "C"]
    queue = asyncio.Queue()
    tasks = [asyncio.create_task(async_task(name, 1, queue, i)) for i, name in enumerate(task_names)]

    finished_set = set()
    total_tasks = len(tasks)
    all_results: List[Dict] = []

    while len(finished_set) < total_tasks:
        item = await queue.get()
        if stream:
            yield item
        else:
            all_results.append(item)

        if item.get("done"):
            finished_set.add(item["index"])

    if not stream:
        yield all_results

# ---------------------------
# 演示调用
# ---------------------------
async def run_demo(stream: bool = True):
    start = time.perf_counter()
    
    gen = _task_generator(stream=stream)
    
    if stream:
        async for v in gen:
            print(f"[Stream] index={v['index']}: {v['text']}")
    else:
        # 非流式，直接 await 生成器获取列表
        all_results = await gen.__anext__()
        print("[Batch] Generated text:")
        for v in all_results:
            print(f"index={v['index']}: {v['text']}")
    
    end = time.perf_counter()
    print(f"Time elapsed: {end - start:.2f}s")

if __name__ == "__main__":
    import sys
    mode = "stream" if len(sys.argv) > 1 and sys.argv[1] == "stream" else "nonstream"
    asyncio.run(run_demo(stream=(mode == "stream")))
