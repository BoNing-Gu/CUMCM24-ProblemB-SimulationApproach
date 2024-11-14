import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

import pickle

# 从文件加载模型
with open('./model/cubic_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 封装转换器函数
def convert_p_sampling_to_p_real(p_sampling_input):
    # 使用加载的模型进行转换
    return loaded_model(p_sampling_input)

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

class Particle:
    def __init__(self, dim: int):
        self.position = np.random.randint(2, size=dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.best_position = self.position.copy()
        self.best_score = float('-inf')


def fitness(particle: Particle, row: pd.Series, num_simulations: int) -> float:
    inspect_comp1, inspect_comp2, inspect_comp3, inspect_comp4, inspect_comp5, inspect_comp6, inspect_comp7, inspect_comp8, inspect_half1, inspect_half2, inspect_half3, inspect_product, disassemble = particle.position

    component_params = [
        [particle.position[i], convert_p_sampling_to_p_real(row[f'c{i + 1}_defect_rate']), row[f'c{i + 1}_purchase_price'],
         row[f'c{i + 1}_inspection_cost']]
        for i in range(8)
    ]

    half_params = [
        [particle.position[i + 8], convert_p_sampling_to_p_real(row[f'h{i + 1}_defect_rate']), row[f'h{i + 1}_assembly_cost'],
         row[f'h{i + 1}_inspection_cost'], row[f'h{i + 1}_disassembly_cost']]
        for i in range(3)
    ]

    p_params = [inspect_product, disassemble, convert_p_sampling_to_p_real(row['p_defect_rate']), row['p_assembly_cost'], row['p_inspection_cost'],
                row['p_market_price'], row['p_replacement_loss'], row['p_disassembly_cost']]

    profits = [simulation(component_params, half_params, p_params[:]) for _ in range(num_simulations)]
    return sum(profits) / num_simulations


def pso_algorithm(row: pd.Series, num_particles: int, num_iterations: int, num_simulations: int) -> Tuple[
    np.ndarray, float, List[float]]:
    dim = 13  # 13 binary decision variables
    particles = [Particle(dim) for _ in range(num_particles)]
    global_best_position = np.zeros(dim)
    global_best_score = float('-inf')

    w = 0.7  # inertia weight
    c1 = 1.5  # cognitive weight
    c2 = 1.5  # social weight

    history = []

    for iteration in range(num_iterations):
        for particle in particles:
            score = fitness(particle, row, num_simulations)

            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()

            if score > global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()

        for particle in particles:
            r1, r2 = np.random.rand(2)
            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (global_best_position - particle.position))

            particle.position = np.clip(particle.position + particle.velocity, 0, 1).round().astype(int)

        history.append(global_best_score)
        print(f"Iteration {iteration + 1}: Best fitness = {global_best_score}")

    return global_best_position, global_best_score, history


def plot_fitness_history(history: List[float], case_num: int):
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title(f'Fitness History - PSO (Case {case_num})')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.savefig(f'./fig/Q4/fitness_history_PSO_Case_{case_num}.svg')
    plt.close()


# 主程序
df = pd.read_excel('./data/Q3/Q3.xlsx')
output = pd.DataFrame()

for i, row in df.iterrows():
    case_num = int(row['case'])
    print(f'-------Case {case_num}-------')

    st = time.time()

    best_strategy, best_profit, history = pso_algorithm(row, num_particles=100, num_iterations=20,
                                                        num_simulations=100000)

    strategy_string = (
            f"零件1: {best_strategy[0]}, 零件2: {best_strategy[1]}, 零件3: {best_strategy[2]}, 零件4: {best_strategy[3]}, \n" +
            f"零件5: {best_strategy[4]}, 零件6: {best_strategy[5]}, 零件7: {best_strategy[6]}, 零件8: {best_strategy[7]}, \n" +
            f"半成品1: {best_strategy[8]}, 半成品2: {best_strategy[9]}, 半成品3: {best_strategy[10]}, \n" +
            f"产品是否检查：{best_strategy[11]}, 次品是否拆解：{best_strategy[12]}")

    print(strategy_string)
    print(f"最佳平均利润: {best_profit:.4f}")

    ed = time.time()
    print(f"运行时间: {ed - st:.2f} 秒")

    plot_fitness_history(history, case_num)
    new_row = pd.DataFrame({'策略': [strategy_string], f'case{case_num}': [best_profit]})
    output = pd.concat([output, new_row], ignore_index=True)

output.to_excel('./data/Q4/Q4_output_Q3update.xlsx', index=False)