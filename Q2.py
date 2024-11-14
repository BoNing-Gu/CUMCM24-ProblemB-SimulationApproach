import pandas as pd
import numpy as np
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

    def sell(self):
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


def simulation(c_params, p_param):
    # params = [
    #     [True, 0.1, 4, 2],
    #     [True, 0.1, 18, 3]
    # ]
    components = []
    for param in c_params:
        component = Component(param[0], param[1], param[2], param[3])
        if component.CHECK:
            component.check()
        components.append(component)

    # param = [True, True, 0.1, 6, 3, 56, 6, 5, components]
    p_param.append(components)
    product = Product(p_param[0], p_param[1], p_param[2], p_param[3], p_param[4], p_param[5], p_param[6], p_param[7], p_param[8])
    if product.CHECK:
        product.check()
    revenue = product.sell()
    for component in product.components:
        revenue += component.count()
    return revenue


states = [True, False]
num_simulations = 100000

def plot_revenue_distirbute(case_df, case_num):
    plt.figure(figsize=(10, 10))
    sns.boxplot(
        data=case_df,
        showfliers=False,  # 剔除离群值
        meanline=True,  # 绘制均值线
        showmeans=True,  # 显示均值点或线
        meanprops=dict(color='blue', linestyle='--', linewidth=2),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='purple', linewidth=1.5),
        capprops=dict(color='green', linewidth=2)
    )
    plt.title(f'Case {case_num} 各决策利润分布')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'./fig/Q2/Case_{case_num}_profit_distribution_noshowfliers.svg')
    plt.close()

df = pd.read_excel('./data/Q2/Q2.xlsx')
output = pd.DataFrame()
for i, row in df.iterrows():
    output_col = []
    case_df = pd.DataFrame()
    case_num = int(row['case'])
    print(f'-------Case {case_num}-------')

    for inspect_comp1 in states:
        for inspect_comp2 in states:
            for inspect_product in states:
                for disassemble in states:
                    c1_params = [inspect_comp1, row['c1_defect_rate'], row['c1_purchase_price'], row['c1_inspection_cost']]
                    c2_params = [inspect_comp2, row['c2_defect_rate'], row['c2_purchase_price'], row['c2_inspection_cost']]
                    p_params = [inspect_product, disassemble, row['p_defect_rate'], row['p_assembly_cost'], row['p_inspection_cost'], row['p_market_price'], row['p_replacement_loss'], row['p_disassembly_cost']]
                    print(c1_params)
                    print(c2_params)
                    print(p_params)

                    profits = []
                    for _ in range(num_simulations):
                        profit = simulation([c1_params, c2_params], p_params[:])
                        profits.append(profit)
                    print(len(profits))

                    string = f"零部件1是否检查: {inspect_comp1}, 零部件2是否检查: {inspect_comp2}, 产品是否检查：{inspect_product}, 次品是否拆解：{disassemble}"
                    print(string)
                    average_profit = sum(profits) / num_simulations
                    print(f"平均利润: {average_profit:.4f}")

                    output_col.append(average_profit)
                    case_df[string] = profits

    output_col = pd.DataFrame(output_col, columns=[f'case{int(row['case'])}'])
    output = pd.concat([output, output_col], axis=1)
    plot_revenue_distirbute(case_df, case_num)

output.to_excel('./data/Q2/Q2_output_noshowfliers.xlsx', index=False)