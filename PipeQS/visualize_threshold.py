import pandas as pd
import matplotlib.pyplot as plt
import os
READ_CSV = "training_results.csv"


def plot_computation_communication_reduce_time(df, aggregation='mean'):
    """
    绘制计算时间、通信时间和规约时间的柱状图，按method、bits、stale_label、pipeline_label区分。

    参数:
    df (DataFrame): 包含实验结果的数据框
    aggregation (str): 指定如何处理相同method、bits、stale的数据，'mean'取平均值，'max'取最大值，'min'取最小值
    """
    # 转换Stale和Pipeline列为更具描述性的标签
    df['stale_label'] = df['stale'].apply(lambda x: 'Stale' if x else 'No_Stale')
    df['pipeline_label'] = df['enable_pipeline'].apply(lambda x: 'Pipeline' if x else 'No_Pipeline')
    
    # 处理Stale Threshold，添加到stale_label
    df['stale_threshold_label'] = df.apply(lambda x: f"Stale_Thresh_{x['stale_threshold']}" if x['stale'] else 'No_Stale', axis=1)
    
    # 按method, bits, stale_label, pipeline_label, stale_threshold_label分组，并根据给定的聚合方法计算时间
    aggregation_funcs = {
        'mean': 'mean',
        'max': 'max',
        'min': 'min'
    }
    
    grouped = df.groupby(['dataset', 'partition', 'partition-method', 'model', 'method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label']).agg({
        'computation': aggregation_funcs[aggregation],
        'communication': aggregation_funcs[aggregation],
        'reduce': aggregation_funcs[aggregation]
    }).reset_index()
    
    # 为每个不同的组合生成一个图表
    for (dataset, partition, partition_method, model), group in grouped.groupby(['dataset', 'partition', 'partition-method', 'model']):
        group_name = f"{dataset}_{partition}_{partition_method}_{model}_{aggregation}"
        group = group.set_index(['method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label'])
        
        ax = group.plot(kind='bar', stacked=True, figsize=(14, 7), y=['computation', 'communication', 'reduce'], color=['#4e79a7', '#f28e2c', '#76b7b2'])
        
        # 为每个部分添加数值标签，显示在中央，并在柱子顶部添加总时间
        for i, (index, row) in enumerate(group.iterrows()):
            computation = row['computation']
            communication = row['communication']
            reduce = row['reduce']
            total_time = computation + communication + reduce
            bottom_computation = computation / 2
            bottom_communication = computation + (communication / 2)
            bottom_reduce = computation + communication + (reduce / 2)

            ax.text(i, bottom_computation, f"{computation:.2f}", ha='center', va='center', color='white', fontsize=9)
            ax.text(i, bottom_communication, f"{communication:.2f}", ha='center', va='center', color='black', fontsize=9)
            ax.text(i, bottom_reduce, f"{reduce:.2f}", ha='center', va='center', color='black', fontsize=9)
            ax.text(i, total_time + 0.005, f"{total_time:.2f}", ha='center', va='bottom', color='black', fontsize=10, weight='bold', rotation=0)
        
        plt.title(f"Computation, Communication, Reduce Time for {dataset}, Partition {partition}, {partition_method}, Model {model}")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Method, Bits, Stale, Pipeline, Stale Threshold")
        plt.xticks(fontsize=12, rotation=45, ha='right')  # 斜着显示横轴标签
        plt.yticks(fontsize=12)  # 增大纵轴刻度字体大小
        plt.legend(title="Method, Bits, Stale, Pipeline, Stale Threshold", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图表
        output_dir = 'charts'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{group_name}.png"))
        plt.close()

def plot_accuracy_vs_epoch(df, selection='all'):
    """
    绘制准确率随epoch变化的折线图，按method、bits、stale_label、pipeline_label、stale_threshold_label、unique_id区分。

    参数:
    df (DataFrame): 包含实验结果的数据框
    selection (str): 指定如何选择数据绘制图表，'all'绘制所有，'max_epoch'绘制epoch数量最多的
    """
    # 转换Stale和Pipeline列为更具描述性的标签
    df['stale_label'] = df['stale'].apply(lambda x: 'Stale' if x else 'No_Stale')
    df['pipeline_label'] = df['enable_pipeline'].apply(lambda x: 'Pipeline' if x else 'No_Pipeline')
    
    # 处理Stale Threshold，添加到stale_label
    df['stale_threshold_label'] = df.apply(lambda x: f"Stale_Thresh_{x['stale_threshold']}" if x['stale'] else 'No_Stale', axis=1)
    
    # 为每个不同的组合生成一个图表
    for (dataset, partition, partition_method, model), group in df.groupby(['dataset', 'partition', 'partition-method', 'model']):
        group_name = f"line_{dataset}_{partition}_{partition_method}_{model}"
        
        fig, ax = plt.subplots(figsize=(14, 7))  # 调整图表宽度
        for (method, bits, stale_label, pipeline_label, stale_threshold_label), subgroup in group.groupby(['method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label']):
            if selection == 'max_epoch':
                # 选择记录epoch数量最多的unique_id
                max_epochs_id = subgroup.groupby('unique_id')['epoch'].count().idxmax()
                subgroup = subgroup[subgroup['unique_id'] == max_epochs_id]
            # 绘制每个unique_id的数据
            for unique_id, data in subgroup.groupby('unique_id'):
                ax.plot(data['epoch'], data['accuracy'], marker='o', label=f"{method}, {bits}, {stale_label}, {pipeline_label}, {stale_threshold_label}, ID={unique_id}")
        
        plt.title(f"Accuracy vs Epoch for {dataset}, Partition {partition}, {partition_method}, Model {model}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.xticks(fontsize=12, rotation=45, ha='right')  # 斜着显示横轴标签
        plt.yticks(fontsize=12)  # 增大纵轴刻度字体大小
        plt.legend(title="Method, Bits, Stale, Pipeline, Stale Threshold, ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图表
        output_dir = 'charts'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{group_name}_{selection}.png"))
        plt.close()


def plot_accuracy_vs_total_time(df, selection='all'):
    """
    绘制准确率随总时间变化的折线图，按method、bits、stale_label、pipeline_label、stale_threshold_label、unique_id区分。

    参数:
    df (DataFrame): 包含实验结果的数据框
    selection (str): 指定如何选择数据绘制图表，'all'绘制所有，'max_epoch'绘制记录的epoch数量最多的
    """
    # 转换Stale和Pipeline列为更具描述性的标签
    df['stale_label'] = df['stale'].apply(lambda x: 'Stale' if x else 'No_Stale')
    df['pipeline_label'] = df['enable_pipeline'].apply(lambda x: 'Pipeline' if x else 'No_Pipeline')
    
    # 处理Stale Threshold，添加到stale_label
    df['stale_threshold_label'] = df.apply(lambda x: f"Stale_Thresh_{x['stale_threshold']}" if x['stale'] else 'No_Stale', axis=1)
    
    # 计算总时间
    df['total_time'] = (df['computation'] + df['communication'] + df['reduce']) * 10
    df['cumulative_time'] = df.groupby('unique_id')['total_time'].cumsum()
    
    # 为每个不同的组合生成一个图表
    for (dataset, partition, partition_method, model), group in df.groupby(['dataset', 'partition', 'partition-method', 'model']):
        group_name = f"time_line_{dataset}_{partition}_{partition_method}_{model}"
        
        fig, ax = plt.subplots(figsize=(14, 7))  # 调整图表宽度
        for (method, bits, stale_label, pipeline_label, stale_threshold_label), subgroup in group.groupby(['method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label']):
            if selection == 'max_epoch':
                # 选择记录epoch数量最多的unique_id
                max_epochs_id = subgroup.groupby('unique_id')['epoch'].count().idxmax()
                subgroup = subgroup[subgroup['unique_id'] == max_epochs_id]
            # 绘制每个unique_id的数据
            for unique_id, data in subgroup.groupby('unique_id'):
                ax.plot(data['cumulative_time'], data['accuracy'], marker='o', label=f"{method}, {bits}, {stale_label}, {pipeline_label}, {stale_threshold_label}, ID={unique_id}")
        
        plt.title(f"Accuracy vs Total Time for {dataset}, Partition {partition}, {partition_method}, Model {model}")
        plt.xlabel("Total Time (seconds)")
        plt.ylabel("Accuracy")
        plt.xticks(fontsize=12, rotation=45, ha='right')  # 斜着显示横轴标签
        plt.yticks(fontsize=12)  # 增大纵轴刻度字体大小
        plt.legend(title="Method, Bits, Stale, Pipeline, Stale Threshold, ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图表
        output_dir = 'charts'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{group_name}_{selection}.png"))
        plt.close()

def plot_communication_times(df, aggregation='mean'):
    """
    绘制qt_time, dq_time, pack_time, unpack_time, true_comm_time, forward_comm_time, backward_comm_time的柱状图。

    参数:
    df (DataFrame): 包含实验结果的数据框
    aggregation (str): 指定如何处理相同method、bits、stale的数据，'mean'取平均值，'max'取最大值，'min'取最小值
    """
    # 转换Stale和Pipeline列为更具描述性的标签
    df['stale_label'] = df['stale'].apply(lambda x: 'Stale' if x else 'No_Stale')
    df['pipeline_label'] = df['enable_pipeline'].apply(lambda x: 'Pipeline' if x else 'No_Pipeline')
    
    # 处理Stale Threshold，添加到stale_label
    df['stale_threshold_label'] = df.apply(lambda x: f"Stale_Thresh_{x['stale_threshold']}" if x['stale'] else 'No_Stale', axis=1)
    
    # 按method, bits, stale_label, pipeline_label, stale_threshold_label分组，并根据给定的聚合方法计算时间
    aggregation_funcs = {
        'mean': 'mean',
        'max': 'max',
        'min': 'min'
    }
    
    grouped = df.groupby(['dataset', 'partition', 'partition-method', 'model', 'method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label']).agg({
        'qt_time': aggregation_funcs[aggregation],
        'dq_time': aggregation_funcs[aggregation],
        'pack_time': aggregation_funcs[aggregation],
        'unpack_time': aggregation_funcs[aggregation],
        'forward_comm_time': aggregation_funcs[aggregation],
        'backward_comm_time': aggregation_funcs[aggregation],
        'true_comm_time': aggregation_funcs[aggregation]
    }).reset_index()
    
    # 为每个不同的组合生成一个图表
    for (dataset, partition, partition_method, model), group in grouped.groupby(['dataset', 'partition', 'partition-method', 'model']):
        group_name = f"comm_time_{dataset}_{partition}_{partition_method}_{model}_{aggregation}"
        fig, ax = plt.subplots(figsize=(14, 7))
        
        index = range(len(group))
        bar_width = 0.25
        
        # 第一个柱子：qt_time, dq_time, pack_time, unpack_time
        p1 = ax.bar(index, group['qt_time'], bar_width, label='QT Time')
        p2 = ax.bar(index, group['dq_time'], bar_width, bottom=group['qt_time'], label='DQ Time')
        p3 = ax.bar(index, group['pack_time'], bar_width, bottom=group['qt_time'] + group['dq_time'], label='Pack Time')
        p4 = ax.bar(index, group['unpack_time'], bar_width, bottom=group['qt_time'] + group['dq_time'] + group['pack_time'], label='Unpack Time')
        
        # 第二个柱子：forward_comm_time, backward_comm_time
        p5 = ax.bar([i + bar_width for i in index], group['forward_comm_time'], bar_width, label='Forward Comm Time')
        p6 = ax.bar([i + bar_width for i in index], group['backward_comm_time'], bar_width, bottom=group['forward_comm_time'], label='Backward Comm Time')
        
        # 第三个柱子：true_comm_time (显示为comm_time)
        p7 = ax.bar([i + 2 * bar_width for i in index], group['true_comm_time'], bar_width, label='Comm Time')
        
        # 为每个部分添加数值标签
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{height:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='center', fontsize=8)

        autolabel(p1)
        autolabel(p2)
        autolabel(p3)
        autolabel(p4)
        autolabel(p5)
        autolabel(p6)
        autolabel(p7)

        plt.title(f"Communication Times for {dataset}, Partition {partition}, {partition_method}, Model {model}")
        plt.ylabel("Time (seconds)")
        plt.xlabel("Method, Bits, Stale, Pipeline, Stale Threshold")
        plt.xticks([i + bar_width for i in index], group['method'] + ', ' + group['bits'].astype(str) + ', ' + group['stale_label'] + ', ' + group['pipeline_label'] + ', ' + group['stale_threshold_label'], rotation=45, ha='right')
        plt.legend(loc='best')
        plt.tight_layout()
        
        # 保存图表
        output_dir = 'charts'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{group_name}.png"))
        plt.close()

def plot_transfer_num_vs_epoch(df, num_column, title_suffix):
    """
    绘制给定的传输数量列与epoch的折线图。

    参数:
    df (DataFrame): 包含实验结果的数据框
    num_column (str): 要绘制的传输数量列的名称
    title_suffix (str): 图表标题的后缀，用于描述绘制的内容
    """
    # 转换Stale和Pipeline列为更具描述性的标签
    df['stale_label'] = df['stale'].apply(lambda x: 'Stale' if x else 'No_Stale')
    df['pipeline_label'] = df['enable_pipeline'].apply(lambda x: 'Pipeline' if x else 'No_Pipeline')
    
    # 处理Stale Threshold，添加到stale_label
    df['stale_threshold_label'] = df.apply(lambda x: f"Stale_Thresh_{x['stale_threshold']}" if x['stale'] else 'No_Stale', axis=1)
    
    # 为每个不同的组合生成一个图表
    for (dataset, partition, partition_method, model), group in df.groupby(['dataset', 'partition', 'partition-method', 'model']):
        group_name = f"{title_suffix}_{dataset}_{partition}_{partition_method}_{model}"
        
        fig, ax = plt.subplots(figsize=(14, 7))
        for (method, bits, stale_label, pipeline_label, stale_threshold_label), subgroup in group.groupby(['method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label']):
            # 绘制每个unique_id的数据
            for unique_id, data in subgroup.groupby('unique_id'):
                ax.plot(data['epoch'], data[num_column], marker='o', label=f"{method}, {bits}, {stale_label}, {pipeline_label}, {stale_threshold_label}, ID={unique_id}")
        
        plt.title(f"{title_suffix} vs Epoch for {dataset}, Partition {partition}, {partition_method}, Model {model}")
        plt.xlabel("Epoch")
        plt.ylabel(title_suffix)
        plt.xticks(fontsize=12, rotation=45, ha='right')
        plt.yticks(fontsize=12)
        plt.legend(title="Method, Bits, Stale, Pipeline, Stale Threshold, ID", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存图表
        output_dir = 'charts'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{group_name}.png"))
        plt.close()


def generate_accuracy_table(df, aggregation='max'):
    """
    生成记录每种方法最高准确率的可视化表格。

    参数:
    df (DataFrame): 包含实验结果的数据框
    aggregation (str): 指定如何处理相同method、bits、stale的数据，'max'取最大值，'mean'取平均值
    """
    # 转换Stale和Pipeline列为更具描述性的标签
    df['stale_label'] = df['stale'].apply(lambda x: 'Stale' if x else 'No_Stale')
    df['pipeline_label'] = df['enable_pipeline'].apply(lambda x: 'Pipeline' if x else 'No_Pipeline')
    
    # 处理Stale Threshold，添加到stale_label
    df['stale_threshold_label'] = df.apply(lambda x: f"Stale_Thresh_{x['stale_threshold']}" if x['stale'] else 'No_Stale', axis=1)
    
    # 计算最高准确率和出现的时间
    grouped = df.groupby(['dataset', 'partition', 'partition-method', 'model', 'method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label', 'unique_id']).apply(lambda x: x.loc[x['accuracy'].idxmax()]).reset_index(drop=True)
    
    # 再次分组以获得最终的最高或平均准确率
    aggregation_funcs = {
        'max': 'max',
        'mean': 'mean'
    }
    
    final_group = grouped.groupby(['dataset', 'partition', 'partition-method', 'model', 'method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label']).agg({
        'accuracy': aggregation_funcs[aggregation],
        'epoch': 'max'  # 假设我们要显示出现的时间
    }).reset_index()

    # 按不同组合生成表格
    for (dataset, partition, partition_method, model), group in final_group.groupby(['dataset', 'partition', 'partition-method', 'model']):
        group_name = f"accuracy_table_{dataset}_{partition}_{partition_method}_{model}_{aggregation}"
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        table_data = group[['method', 'bits', 'stale_label', 'pipeline_label', 'stale_threshold_label', 'accuracy', 'epoch']]
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(table_data.columns))))
        
        plt.title(f"Highest Accuracy by Method for {dataset}, Partition {partition}, {partition_method}, Model {model}")
        
        # 保存表格图片
        output_dir = 'charts'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"{group_name}.png"), bbox_inches='tight')
        plt.close()


def main():
    # 读取CSV文件
    csv_file = READ_CSV
    df = pd.read_csv(csv_file)
    
    # 调用图表生成函数
    # plot_computation_communication_reduce_time(df, aggregation='mean')
    # plot_computation_communication_reduce_time(df, aggregation='max')
    plot_computation_communication_reduce_time(df, aggregation='min')

    # 调用图表生成函数
    # plot_accuracy_vs_epoch(df, selection='all')
    plot_accuracy_vs_epoch(df, selection='max_epoch')

    # 调用图表生成函数
    # plot_accuracy_vs_total_time(df, selection='all')
    plot_accuracy_vs_total_time(df, selection='max_epoch')

    # 调用表格生成函数
    generate_accuracy_table(df, aggregation='max')
    # generate_accuracy_table(df, aggregation='mean')

    # 调用图表生成函数
    # plot_communication_times(df, aggregation='mean')
    # plot_communication_times(df, aggregation='max')
    plot_communication_times(df, aggregation='min')

    # 调用图表生成函数
    plot_transfer_num_vs_epoch(df, 'transfer_num', 'Transfer Number')
    plot_transfer_num_vs_epoch(df, 'forward_transfer_num', 'Forward Transfer Number')
    plot_transfer_num_vs_epoch(df, 'backward_transfer_num', 'Backward Transfer Number')

if __name__ == "__main__":
    main()
