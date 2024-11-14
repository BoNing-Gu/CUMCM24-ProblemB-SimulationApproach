import pandas as pd
import numpy as np
import time
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

class Component:  # 零部件
    def __init__(self, CHECK, defect_rate, purchase_price, inspection_cost):
        self.cost = 0                              # 成本
        self.defect_rate = defect_rate             # 次品率
        self.purchase_price = purchase_price       # 进件成本
        self.inspection_cost = inspection_cost     # 检测费用
        self.defect = False                        # 是否是次品
        self.CHECK = CHECK                         # 是否检查零件
        self.new()

    def new(self):   # 重置新零件
        self.cost += self.purchase_price
        self.defect = np.random.random() < self.defect_rate

    def check(self):  # 检查零件
        self.cost += self.inspection_cost
        while self.defect:  # 如果是次品就替换，直到找到合格品
            self.new()
            self.cost += self.inspection_cost

    def count(self):  # 结算利润
        return 0 - self.cost

class HalfProduct:  # 半成品
    def __init__(self, CHECK, defect_rate, assembly_cost, inspection_cost, disassembly_cost, components):
        self.cost = 0
        self.components = components  # 零件列表
        self.defect_rate = defect_rate  # 次品率
        self.assembly_cost = assembly_cost  # 装配成本
        self.inspection_cost = inspection_cost  # 检测费用
        self.disassembly_cost = disassembly_cost  # 拆解费用
        self.defect = False  # 是否是次品
        self.CHECK = CHECK  # 是否检查产品
        # self.DISASSEMBLE = DISASSEMBLE  # 简化问题，一定拆解零件，不考虑丢弃
        self.DROP = False
        self.assemble()

    def assemble(self):  # 组装
        self.defect = False
        self.cost += self.assembly_cost
        for component in self.components:
            if component.defect:  # 只要有零件不合格，产品一定不合格
                self.defect = True
        if not self.defect:  # 即使零件合格，产品也可能不合格
            self.defect = np.random.random() < self.defect_rate
            # bool(np.random.binomial(1, self.defect_rate, 1))

    # def drop(self):  # 丢弃产品
    #     self.DROP = True

    def disassemble(self):  # 拆解产品
        self.cost += self.disassembly_cost
        for component in self.components:
            component.check()
        self.assemble()

    def check(self):  # 检查产品
        self.cost += self.inspection_cost
        if self.defect:  # 如果是次品就拆解
            # if self.DISASSEMBLE:
            self.disassemble()
            self.check()
            # else:
            #     self.drop()

    def count(self):  # 结算利润
        return 0 - self.cost

class Product:
    def __init__(self, CHECK, DISASSEMBLE, defect_rate, assembly_cost, inspection_cost, market_price, replacement_loss, disassembly_cost, components):
        self.cost = 0
        self.revenue = 0
        self.components = components              # 零件列表
        self.defect_rate = defect_rate            # 次品率
        self.assembly_cost = assembly_cost        # 装配成本
        self.inspection_cost = inspection_cost    # 检测费用
        self.market_price = market_price          # 售价
        self.replacement_loss = replacement_loss  # 调换成本
        self.disassembly_cost = disassembly_cost  # 拆解费用
        self.defect = False                       # 是否是次品
        self.CHECK = CHECK                        # 是否检查产品
        self.DISASSEMBLE = DISASSEMBLE            # 是否拆解零件
        self.DROP = False
        self.assemble()

    def assemble(self):  # 组装
        self.defect = False
        self.cost += self.assembly_cost
        for component in self.components:
            if component.defect:      # 只要有零件不合格，产品一定不合格
                self.defect = True
        if not self.defect:           # 即使零件合格，产品也可能不合格
            self.defect = np.random.random() < self.defect_rate
                # bool(np.random.binomial(1, self.defect_rate, 1))

    def drop(self):      # 丢弃产品
        self.DROP = True

    def disassemble(self):   # 拆解产品
        self.cost += self.disassembly_cost
        for component in self.components:
            component.check()
        self.assemble()

    def check(self):     # 检查产品
        self.cost += self.inspection_cost
        if self.defect:  # 如果是次品就拆解或者丢弃
            if self.DISASSEMBLE:
                self.disassemble()
                self.check()
            else:
                self.drop()

    def sell(self):    # 出售产品
        if self.DROP:
            return self.count()
        else:
            self.revenue += self.market_price
            if self.defect:  # 如果是次品要无条件调换
                self.cost += self.replacement_loss
                self.disassemble()
                self.check()
            return self.count()

    def count(self):  # 结算利润
        return self.revenue - self.cost


def simulation(c_params, h_params, p_param):
    components = []
    for param in c_params:
        component = Component(param[0], param[1], param[2], param[3])
        if component.CHECK:
            component.check()
        components.append(component)

    half_products = []
    s = 0
    e = 3
    for param in h_params:
        if e == 9:
            e = 8  # 最后一次 6:8
        half_product = HalfProduct(param[0], param[1], param[2], param[3], param[4], components[s:e])
        if half_product.CHECK:
            half_product.check()
        half_products.append(half_product)
        s += 3
        e += 3

    product = Product(p_param[0], p_param[1], p_param[2], p_param[3], p_param[4], p_param[5], p_param[6], p_param[7], half_products)
    if product.CHECK:
        product.check()
    revenue = product.sell()
    for half_product in product.components:
        revenue += half_product.count()
        for component in half_product.components:
            revenue += component.count()
    return revenue

def plot_revenue_distirbute(case_df, case_num):
    plt.figure(figsize=(20, 20))
    sns.boxplot(data=case_df)
    plt.title(f'Case {case_num} 各决策利润分布')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'./fig/Q3/Case_{case_num}_profit_distribution.png')
    plt.close()

def initialize_population(size):
    # 初始化种群：创建一个随机的二进制矩阵，每行代表一个个体，每列代表一个决策变量
    return np.random.randint(2, size=(size, 13))

def fitness(individual, row, num_simulations):
    # 计算适应度（平均利润）
    # 将个体的二进制编码转换为具体的决策变量
    inspect_comp1, inspect_comp2, inspect_comp3, inspect_comp4, inspect_comp5, inspect_comp6, inspect_comp7, inspect_comp8, inspect_half1, inspect_half2, inspect_half3, inspect_product, disassemble = individual

    # 构建组件参数列表
    component_params = []
    for i in range(1, 9):
        component_params.append([
            individual[i - 1], row[f'c{i}_defect_rate'], row[f'c{i}_purchase_price'], row[f'c{i}_inspection_cost']
        ])

    # 构建半成品参数列表
    half_params = []
    for i in range(1, 4):
        half_params.append([
            individual[i + 7], row[f'h{i}_defect_rate'], row[f'h{i}_assembly_cost'], row[f'h{i}_inspection_cost'],
            row[f'h{i}_disassembly_cost']
        ])

    # 构建产品参数列表
    p_params = [inspect_product, disassemble, row['p_defect_rate'], row['p_assembly_cost'], row['p_inspection_cost'],
                row['p_market_price'], row['p_replacement_loss'], row['p_disassembly_cost']]

    # 进行多次模拟并计算平均利润
    profits = []
    for _ in range(num_simulations):
        profit = simulation(component_params, half_params, p_params[:])
        profits.append(profit)

    return sum(profits) / num_simulations


def crossover(parent1, parent2):
    # 交叉操作：在随机位置将两个父代切分并交换后半部分
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutate(individual):
    # 变异操作：以一定概率翻转每个基因
    for i in range(len(individual)):
        if np.random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]
    return individual


def roulette_wheel_selection(population, fitness_scores, num_parents):
    # 轮盘赌选择
    # 确保所有适应度值为非负
    adjusted_fitness = fitness_scores - np.min(fitness_scores)
    # 处理所有适应度值都为0的情况
    if np.sum(adjusted_fitness) == 0:
        return np.random.choice(len(population), size=num_parents, replace=False)
    # 计算选择概率
    probabilities = adjusted_fitness / np.sum(adjusted_fitness)
    # 使用np.random.choice进行选择
    parents_indices = np.random.choice(
        range(len(population)),
        size=num_parents,
        replace=True,  # 允许重复选择
        p=probabilities
    )
    return parents_indices

def plot_fitness_history(history):
    # 跟踪适应度
    plt.figure(figsize=(10, 6))
    plt.plot(history['max'], label='Best Fitness')
    plt.plot(history['avg'], label='Average Fitness')
    plt.plot(history['min'], label='Worst Fitness')
    plt.title(f'Fitness History - Roulette Wheel Selection')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./fig/Q3/fitness_history_Roulette_Wheel_Selection.svg')
    plt.close()

def genetic_algorithm(row, num_simulations):
    # 初始化种群
    population = initialize_population(POPULATION_SIZE)

    # 用于记录每代的适应度数据
    history = {
        'max': [],
        'avg': [],
        'min': []
    }

    # 进化
    for generation in range(NUM_GENERATIONS):
        # 计算每个个体的适应度
        fitness_scores = [fitness(ind, row, num_simulations) for ind in population]

        # 记录本代的适应度数据
        history['max'].append(np.max(fitness_scores))
        history['avg'].append(np.mean(fitness_scores))
        history['min'].append(np.min(fitness_scores))

        # # 选择操作：排序选择（Top-k Selection）
        # parent_indices = np.argsort(fitness_scores)[-POPULATION_SIZE // 2:]
        # parents = population[parent_indices]

        # 选择操作：轮盘赌（Roulette Wheel Selection）
        parents_indices = roulette_wheel_selection(population, fitness_scores, POPULATION_SIZE // 2)
        parents = population[parents_indices]

        # 生成新的种群
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # 随机选择两个父代
            parent_indices = np.random.choice(len(parents), 2, replace=False)
            parent1, parent2 = parents[parent_indices[0]], parents[parent_indices[1]]
            if np.random.random() < CROSSOVER_RATE:
                # 进行交叉
                child1, child2 = crossover(parent1, parent2)
            else:
                # 不交叉，直接复制父代
                child1, child2 = parent1.copy(), parent2.copy()
            # 对子代进行变异
            new_population.extend([mutate(child1), mutate(child2)])

        # 更新种群
        population = np.array(new_population[:POPULATION_SIZE])

        # 输出当前代的最佳适应度
        best_fitness = max(fitness_scores)
        print(f"Generation {generation}: Best fitness = {best_fitness}")

    # 找出最终种群中的最佳个体
    best_individual = population[np.argmax([fitness(ind, row, num_simulations) for ind in population])]
    return best_individual, fitness(best_individual, row, num_simulations), history

# 遗传算法参数
POPULATION_SIZE = 100  # 种群大小
NUM_GENERATIONS = 20   # 迭代代数
MUTATION_RATE = 0.1    # 变异率
CROSSOVER_RATE = 0.8   # 交叉率
# 整数规划
states = [True, False]
num_simulations = 100000  # 模拟次数

# 主程序
df = pd.read_excel('./data/Q3/Q3.xlsx')
output = pd.DataFrame()

for i, row in df.iterrows():
    case_num = int(row['case'])
    print(f'-------Case {case_num}-------')

    st = time.time()

    best_strategy, best_profit, history = genetic_algorithm(row, num_simulations=num_simulations)

    strategy_string = (
                f"零件1: {best_strategy[0]}, 零件2: {best_strategy[1]}, 零件3: {best_strategy[2]}, 零件4: {best_strategy[3]}, \n" +
                f"零件5: {best_strategy[4]}, 零件6: {best_strategy[5]}, 零件7: {best_strategy[6]}, 零件8: {best_strategy[7]}, \n" +
                f"半成品1: {best_strategy[8]}, 半成品2: {best_strategy[9]}, 半成品3: {best_strategy[10]}, \n" +
                f"产品是否检查：{best_strategy[11]}, 次品是否拆解：{best_strategy[12]}")

    print(strategy_string)
    print(f"最佳平均利润: {best_profit:.4f}")

    ed = time.time()
    print(f"运行时间: {ed - st:.2f} 秒")

    plot_fitness_history(history)
    new_row = pd.DataFrame({'策略': [strategy_string], f'case{case_num}': [best_profit]})
    output = pd.concat([output, new_row], ignore_index=True)

output.to_excel('./data/Q3/Q3_GA_output_20代.xlsx', index=False)