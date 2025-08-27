#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本名称: asyncio_stream_vs_batch.py
=================================

本脚本演示了 Python asyncio 的两种异步任务处理模式：
1. 流式返回 (Streaming Mode)
   - 多个任务并发执行
   - 任务一旦产生结果，立即返回给上层调用
   - 适合实时推理、实时处理等场景

2. 批量返回 (Batch Mode)
   - 顺序执行任务
   - 等待所有任务完成后，一次性返回所有结果
   - 适合需要完整数据集后再处理的场景

运行方式:
---------
直接执行脚本即可：
    python asyncio_stream_vs_batch.py

输出效果:
---------
1. 流式模式: 多个任务交替输出结果，耗时约等于单个任务的最大耗时。
2. 批量模式: 任务按顺序完成，输出集中显示，耗时约等于任务耗时总和。
"""

import asyncio
import time
from typing import AsyncGenerator, List


# ---------------------------
# 异步生成器：模拟分步完成的任务
# ---------------------------
async def async_task(name: str, delay: float) -> AsyncGenerator[str, None]:
    """
    模拟一个异步任务，每隔 delay 秒产生一个结果
    """
    for i in range(3):
        await asyncio.sleep(delay)
        yield f"{name} step {i} @ {time.perf_counter():.2f}"
    yield f"{name} done @ {time.perf_counter():.2f}"


# ---------------------------
# 流式模式：任务并发，结果实时返回
# ---------------------------
async def process_streaming_tasks() -> AsyncGenerator[str, None]:
    """
    并发运行多个生成器任务，一旦有结果产生，立即返回（流式返回）
    """
    generators = [
        async_task("A", 1),
        async_task("B", 1),
        async_task("C", 1),
    ]

    # 初始时，为每个生成器启动获取第一个结果的任务
    task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}

    while task_map:
        # 等待任意一个任务完成
        done, _ = await asyncio.wait(task_map.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            gen = task_map.pop(task)
            try:
                # 获取结果并立即返回
                result = task.result()
                yield result

                # 创建下一个任务，继续获取生成器的下一个值
                next_task = asyncio.create_task(gen.__anext__())
                task_map[next_task] = gen

            except StopAsyncIteration:
                # 生成器已经结束，忽略 StopAsyncIteration
                pass


# ---------------------------
# 批量模式：顺序收集所有结果后一次性返回
# ---------------------------
async def process_batch_tasks() -> List[str]:
    """
    顺序运行多个生成器任务，等待所有任务完成后，一次性返回所有结果
    """
    generators = [
        async_task("A", 1),
        async_task("B", 1),
        async_task("C", 1),
    ]

    results = []
    for gen in generators:
        # 注意：这里是顺序执行，每个生成器必须跑完才能进入下一个
        async for result in gen:
            results.append(result)

    return results


# ---------------------------
# 演示函数
# ---------------------------
async def run_stream_mode():
    """运行流式模式"""
    print("=== 流式模式 ===")
    start_time = time.perf_counter()
    async for response in process_streaming_tasks():
        print(f"收到: {response}")
    print(f"\n流式模式耗时: {time.perf_counter() - start_time:.2f}s")


async def run_batch_mode():
    """运行批量模式"""
    print("=== 批量模式 ===")
    start_time = time.perf_counter()
    results = await process_batch_tasks()
    print("所有结果:")
    for result in results:
        print(f"  {result}")
    print(f"批量模式耗时: {time.perf_counter() - start_time:.2f}s")


async def main():
    """主函数：依次运行两种模式"""
    await run_stream_mode()
    await run_batch_mode()


if __name__ == "__main__":
    asyncio.run(main())
