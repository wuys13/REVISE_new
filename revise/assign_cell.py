import pandas as pd
import numpy as np
from tqdm import tqdm

def assign_cell_types_easy(SVC_obs, cell_contributions, mode="max"):
    """
    Args:
        SVC_obs: DataFrame，包含'spot_name'和'cell_id'列
        cell_contributions: DataFrame，每个spot中存在的cell类型贡献
        mode: 模式，"max"或"random"
        
    Returns:
        SVC_obs: 更新后的SVC_obs，包含'cell_type'列
    """
    SVC_obs = SVC_obs.copy()
    cell_contributions = cell_contributions.copy()
    
    # 确保cell_contributions的index与SVC_obs中的spot_name匹配
    assert set(cell_contributions.index) == set(SVC_obs['spot_name'].unique()), "cell_contributions的index与SVC_obs中的spot_name不匹配"
    
    if 'cell_type' not in SVC_obs.columns:
        SVC_obs['cell_type'] = "Unknown"
    
    # 按spot分组处理
    spot_groups = SVC_obs.groupby('spot_name')
    
    for spot_name in tqdm(cell_contributions.index, desc="Assigning cell types"):
        # 获取当前spot中的所有cell
        spot_cells_df = spot_groups.get_group(spot_name)
        spot_cells = spot_cells_df['cell_id'].values
        
        # 获取当前spot的cell类型贡献
        spot_contributions = cell_contributions.loc[spot_name]
        
        if mode == "max":
            # 找出占比最大的cell类型
            max_type = spot_contributions.idxmax()
            # 为spot中的所有cell分配这个类型
            SVC_obs.loc[spot_cells_df.index, 'cell_type'] = max_type
        
        elif mode == "random":
            # 如果只有一个cell，直接分配占比最大的类型
            if len(spot_cells) == 1:
                max_type = spot_contributions.idxmax()
                SVC_obs.loc[spot_cells_df.index, 'cell_type'] = max_type
            else:
                # 找出前两种cell类型（最多贡献和次多贡献）
                sorted_types = spot_contributions.sort_values(ascending=False).index
                if len(sorted_types) >= 2:
                    max_type = sorted_types[0]
                    second_type = sorted_types[1]
                    
                    # 为大部分cell分配最大类型
                    SVC_obs.loc[spot_cells_df.index, 'cell_type'] = max_type
                    
                    # 随机选择一个cell改为次多类型
                    random_index = np.random.choice(spot_cells_df.index)
                    SVC_obs.loc[random_index, 'cell_type'] = second_type
                else:
                    # 如果只有1种类型，全部分配这个类型
                    max_type = sorted_types[0]
                    SVC_obs.loc[spot_cells_df.index, 'cell_type'] = max_type
        else:
            raise ValueError("mode必须是'max'或'random'")
    
    return SVC_obs
    
def assign_cell_types(SVC_obs, PM_on_cell, spot_cell_distribution):
    """    
    Args:
        SVC_obs: DataFrame，包含'spot_name'和'cell_id'列
        PM_on_cell: DataFrame，每个cell的cell类型概率
        spot_cell_distribution: DataFrame，每个spot中各个cell type的细胞数
        
    Returns:
        SVC_obs: 更新后的SVC_obs，包含'cell_type'列
    """
    SVC_obs = SVC_obs.copy()
    type_list = list(spot_cell_distribution.columns)  # 获取细胞类型列表
    PM_on_cell = PM_on_cell.loc[SVC_obs['cell_id'].values, type_list]
    print(PM_on_cell.shape)

    # 添加新的列来存储cell types
    if 'cell_type' not in SVC_obs.columns:
        SVC_obs['cell_type'] = "Unknown"
    
    # 按spot分组处理
    spot_groups = SVC_obs.groupby('spot_name')
    
    # 为每个spot分配细胞类型
    for spot_name in tqdm(spot_cell_distribution.index, desc="Assigning cell types"):
        # 获取当前spot中的所有cell
        spot_cells_df = spot_groups.get_group(spot_name)
        
        # 确保cell_id作为索引存在于PM_on_cell中
        valid_cells = spot_cells_df[spot_cells_df['cell_id'].isin(PM_on_cell.index)]
        if len(valid_cells) == 0:
            print(f"Warning: No valid cells found for spot {spot_name}")
            # print(spot_cells_df)
            # print(PM_on_cell.head())
            # exit()
            continue
            
        # 获取每个cell对应的cell类型概率
        cell_probs = PM_on_cell.loc[valid_cells['cell_id']].values
        
        # 获取该spot需要的每种类型的细胞数量
        target_counts = spot_cell_distribution.loc[spot_name].astype(int)  # 确保是整数
        
        # 初始分配：每个cell选择最高概率的type
        cell_type_indices = np.argmax(cell_probs, axis=1)
        initial_types = np.array(type_list)[cell_type_indices]
        
        # 统计初始分配的细胞数
        type_counts = pd.Series(0, index=type_list)
        for t in initial_types:
            type_counts[t] += 1
        
        # 创建需要调整的类型列表
        adjustments = []
        for cell_type in type_list:
            target = int(target_counts[cell_type])
            current = type_counts[cell_type]
            if current != target:
                adjustments.append({
                    'cell_type': cell_type,
                    'difference': current - target,  # 正数表示过多，负数表示不足
                    'target': target
                })
        
        # 按照差异的绝对值排序，优先处理差异大的
        adjustments.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        # 处理需要调整的类型
        for adj in adjustments:
            cell_type = adj['cell_type']
            difference = adj['difference']
            
            if difference > 0:  # 需要减少的类型
                # 找到这个类型的所有细胞
                mask = initial_types == cell_type
                cells_of_type = np.where(mask)[0]
                
                if len(cells_of_type) > 0:
                    # 计算这些细胞对其他类型的概率
                    probs = cell_probs[cells_of_type]
                    probs[:, type_list.index(cell_type)] = -np.inf  # 排除当前类型
                    
                    # 选择最适合重新分配的细胞
                    best_alternative_scores = np.max(probs, axis=1)
                    cells_to_change = cells_of_type[np.argsort(best_alternative_scores)[-difference:]]
                    
                    # 为这些细胞重新分配类型
                    for cell_idx in cells_to_change:
                        new_type_idx = np.argmax(cell_probs[cell_idx])
                        initial_types[cell_idx] = type_list[new_type_idx]
                    
            elif difference < 0:  # 需要增加的类型
                # 找到其他类型的细胞
                mask = initial_types != cell_type
                other_cells = np.where(mask)[0]
                
                if len(other_cells) > 0:
                    # 计算这些细胞对当前类型的概率
                    probs = cell_probs[other_cells]
                    type_idx = type_list.index(cell_type)
                    
                    # 选择最适合改为当前类型的细胞
                    best_cells = other_cells[np.argsort(probs[:, type_idx])[-abs(difference):]]
                    
                    # 更新这些细胞的类型
                    initial_types[best_cells] = cell_type
        
        # 更新SVC_obs中的cell types
        SVC_obs.loc[valid_cells.index, 'cell_type'] = initial_types
    
    return SVC_obs