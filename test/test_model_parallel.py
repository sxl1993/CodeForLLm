
world_size = 8
tensor_model_parallel_size = 2
pipeline_model_parallel_size = 4
num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
tp_group_ranks = []
for i in range(num_tensor_model_parallel_groups):
    ranks = list(
        range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
    )
    tp_group_ranks.append(ranks)
    
print(f"tp_group_ranks:", tp_group_ranks)


pp_group_ranks = []
num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
for i in range(num_pipeline_model_parallel_groups):
    ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
    pp_group_ranks.append(ranks)
    
print(f"pp_group_ranks:", pp_group_ranks)