import networkx as nx

from data.node import WrapperGraph, Node


def simple_graph():
    g = nx.DiGraph()
    g.add_edges_from([('a', 'b'), ('a', 'c'), ('b', 'd'), ('b', 'e'), ('c', 'f'), ('c', 'g'), ('e', 'x')])
    return WrapperGraph(g)


def copy_example():
    return [
        # ["MethodDeclaration", 816.2052612304688, 101], ["String", 450.58026123046875, 140],
        #  ["f", 528.5802612304688, 140],
        #  ["Parameter", 606.5802612304688, 140], ["String", 567.5802612304688, 179], ["txt", 645.5802612304688, 179],

        ["BlockStmt", 1181.8302001953125, 140], ["ExpressionStmt", 704.0802612304688, 179],
        ["AssignExpr:assign", 704.0802612304688, 218], ["txt", 665.0802612304688, 257],
        ["MethodCallnExpr", 743.0802612304688, 257], ["replace", 626.0802612304688, 296],
        ["txt", 704.0802612304688, 296],
        ["&", 782.0802612304688, 296], ["&amp;", 860.0802612304688, 296], ["ExpressionStmt", 996.5802612304688, 179],
        ["AssignExpr:assign", 996.5802612304688, 218], ["txt", 957.5802612304688, 257],
        ["MethodCallnExpr", 1035.580322265625, 257], ["replace", 918.5802612304688, 296],
        ["txt", 996.5802612304688, 296],
        ['"', 1074.580322265625, 296], ["&quote;", 1152.5802001953125, 296],
        ["ExpressionStmt", 1289.0802001953125, 179], ["AssignExpr:assign", 1289.0802001953125, 218],
        ["txt", 1250.0802001953125, 257], ["MethodCallnExpr", 1328.0802001953125, 257],
        ["replace", 1211.0802001953125, 296], ["txt", 1289.0802001953125, 296], ["<", 1367.080078125, 296],
        ["&lt;", 1445.080078125, 296], ["ExpressionStmt", 1581.580078125, 179],
        ["AssignExpr:assign", 1581.580078125, 218], ["txt", 1542.580078125, 257],
        ["MethodCallnExpr", 1620.580078125, 257], ["replace", 1503.580078125, 296], ["txt", 1581.580078125, 296],
        [">", 1659.580322265625, 296], ["&gt;", 1737.580322265625, 296], ["ReturnStmt", 1659.580322265625, 179],
        ["txt", 1659.580322265625, 218]]


def example_factorial():
    return [["MethodDeclaration", 704.2348022460938, 71.58443450927734],
            ["int", 606.7348022460938, 101.58443450927734], ["f", 666.7348022460938, 101.58443450927734],
            ["Parameter", 726.7348022460938, 101.58443450927734], ["int", 696.7348022460938, 131.58444213867188],
            ["n", 756.7348022460938, 131.58444213867188], ["BlockStmt", 801.7348022460938, 101.58443450927734],
            ["IfStmt", 801.7348022460938, 131.58444213867188],
            ["BinaryExpr:equals", 734.2348022460938, 161.58444213867188], ["n", 704.2348022460938, 191.58444213867188],
            ["0", 764.2348022460938, 191.58444213867188], ["BlockStmt", 809.2348022460938, 161.58444213867188],
            ["ReturnStmt", 809.2348022460938, 191.58444213867188], ["1", 809.2348022460938, 221.58444213867188],
            ["BlockStmt", 869.2348022460938, 161.58444213867188], ["ReturnStmt", 869.2348022460938, 191.58444213867188],
            ["BinaryExpr:times", 869.2348022460938, 221.58444213867188], ["n", 839.2348022460938, 251.58444213867188],
            ["MethodCallnExpr", 899.2348022460938, 251.58444213867188], ["f", 869.2348022460938, 281.5844421386719],
            ["BinaryExpr:minus", 929.2348022460938, 281.5844421386719], ["n", 899.2348022460938, 311.5844421386719],
            ["1", 959.2348022460938, 311.5844421386719]]


def example_index_of():
    return [
        # ["MethodDeclaration",739.2335205078125,41.8775520324707],["int",649.2335205078125,71.87755584716797],["f",709.2335205078125,71.87755584716797],["Parameter",769.2335205078125,71.87755584716797],
        ["BlockStmt", 829.2335205078125, 71.87755584716797], ["ExpressionStmt", 615.4835205078125, 101.87755584716797],
        ["VariableDeclarationExpr", 615.4835205078125, 131.8775634765625],
        ["int", 585.4835205078125, 161.8775634765625], ["VariableDeclarator", 645.4835205078125, 161.8775634765625],
        ["i", 615.4835205078125, 191.87754821777344], ["0", 675.4835205078125, 191.87754821777344],
        ["ForeachStmt", 834.8585205078125, 101.87755584716797],
        ["VariableDeclarationExpr", 720.4835205078125, 131.8775634765625],
        ["Object", 690.4835205078125, 161.8775634765625], ["VariableDeclarator", 750.4835205078125, 161.8775634765625],
        ["elem", 750.4835205078125, 191.87754821777344], ["FieldnAccessnExpr", 825.4835205078125, 131.8775634765625],
        ["this", 795.4835205078125, 161.8775634765625], ["elements", 855.4835205078125, 161.8775634765625],
        ["BlockStmt", 949.2335205078125, 131.8775634765625], ["IfStmt", 900.4835205078125, 161.8775634765625],
        ["MethodCallnExpr", 847.9835205078125, 191.87754821777344], ["elem", 787.9835205078125, 221.87754821777344],
        ["equals", 847.9835205078125, 221.87754821777344], ["target", 907.9835205078125, 221.87754821777344],
        ["BlockStmt", 952.9835205078125, 191.87754821777344], ["ReturnStmt", 952.9835205078125, 221.87754821777344],
        ["i", 952.9835205078125, 251.87754821777344], ["ExpressionStmt", 997.9835205078125, 161.8775634765625],
        ["UnarynExpr:posnIncrement", 997.9835205078125, 191.87754821777344],
        ["i", 997.9835205078125, 221.87754821777344], ["ReturnStmt", 1042.9835205078125, 101.87755584716797],
        ["UnarynExpr:negative", 1042.9835205078125, 131.8775634765625], ["1", 1042.9835205078125, 161.8775634765625]]


def example_sum():
    return [["MethodDeclaration", 447.2051696777344, 7.000020503997803],
            ["long", 369.2051696777344, 46.000022888183594], ["sum", 447.2051696777344, 46.000022888183594],
            ["BlockStmt", 525.2052001953125, 46.000022888183594],
            ["ExpressionStmt", 228.43954467773438, 85.0000228881836],
            ["VariableDeclarationExpr", 228.43954467773438, 124.0000228881836], ["long", 189.43954467773438, 163],
            ["VariableDeclarator", 267.4395446777344, 163], ["sum", 228.43954467773438, 202],
            ["base", 306.4395751953125, 202], ["ExpressionStmt", 364.9395446777344, 85.0000228881836],
            ["VariableDeclarationExpr", 364.9395446777344, 124.0000228881836], ["Cell", 325.9395751953125, 163],
            ["VariableDeclarator", 403.9395446777344, 163], ["as", 364.9395446777344, 202],
            ["cells", 442.9395446777344, 202], ["IfStmt", 632.4552001953125, 85.0000228881836],
            ["BinaryExpr:notEquals", 501.4395446777344, 124.0000228881836], ["as", 462.4395446777344, 163],
            ["null", 540.4395751953125, 163], ["BlockStmt", 763.4708251953125, 124.0000228881836],
            ["ExpressionStmt", 598.9395751953125, 163], ["VariableDeclarationExpr", 598.9395751953125, 202],
            ["int", 559.9395751953125, 241], ["VariableDeclarator", 637.9395751953125, 241],
            ["n", 598.9395751953125, 280], ["FieldAccessExpr", 676.9395751953125, 280], ["as", 637.9395751953125, 319],
            ["length", 715.9395751953125, 319], ["ForStmt", 928.0020751953125, 163],
            ["VariableDeclarationExpr", 735.4395751953125, 202], ["int", 696.4395751953125, 241],
            ["VariableDeclarator", 774.4395751953125, 241], ["i", 735.4395751953125, 280],
            ["0", 813.4395751953125, 280], ["BinaryExpr:less", 871.9395751953125, 202], ["i", 832.9395751953125, 241],
            ["n", 910.9395751953125, 241], ["UnaryExpr:preIncrement", 969.4395751953125, 202],
            ["i", 969.4395751953125, 241], ["BlockStmt", 1120.5645751953125, 202],
            ["ExpressionStmt", 1027.9395751953125, 241], ["VariableDeclarationExpr", 1027.9395751953125, 280],
            ["Cell", 988.9395751953125, 319], ["VariableDeclarator", 1066.9395751953125, 319],
            ["a", 1027.9395751953125, 358], ["ArrayAccessExpr", 1105.9395751953125, 358],
            ["as", 1066.9395751953125, 397], ["i", 1144.9395751953125, 397], ["IfStmt", 1213.1895751953125, 241],
            ["BinaryExpr:notEquals", 1164.4395751953125, 280], ["a", 1125.4395751953125, 319],
            ["null", 1203.4395751953125, 319], ["ExpressionStmt", 1261.9395751953125, 280],
            ["AssignExpr:plus", 1261.9395751953125, 319], ["sum", 1222.9395751953125, 358],
            ["FieldAccessExpr", 1300.9395751953125, 358], ["a", 1261.9395751953125, 397],
            ["value", 1339.9395751953125, 397], ["ReturnStmt", 821.9708251953125, 85.0000228881836],
            ["sum", 821.9708251953125, 124.0000228881836]]


def example_http_post():
    return [["MethodDeclaration", 462.2052001953125, 67.33332824707031],
            ["void", 91.70524597167969, 106.33332824707031], ["f", 169.7052001953125, 106.33332824707031],
            ["Parameter", 247.7052001953125, 106.33332824707031], ["Parameter", 325.7052001953125, 106.33332824707031],
            ["HttpReadResult", 286.7052001953125, 145.3333282470703], ["result", 364.7052001953125, 145.3333282470703],
            ["ConnectionException", 579.2052001953125, 106.33332824707031],
            ["BlockStmt", 832.7052612304688, 106.33332824707031],
            ["ExpressionStmt", 423.2052001953125, 145.3333282470703],
            ["VariableDeclarationExpr", 423.2052001953125, 184.33331298828125],
            ["ExpressionStmt", 501.2052001953125, 145.3333282470703],
            ["MethodCallExpr", 501.2052001953125, 184.33331298828125],
            ["method", 384.2052001953125, 223.33331298828125], ["setHeader", 462.2052001953125, 223.33331298828125],
            ["User-Agent", 540.2052001953125, 223.33331298828125],
            ["FieldAccessExpr", 618.2052001953125, 223.33331298828125],
            ["HttpConnection", 579.2052001953125, 262.33331298828125],
            ["USER_AGENT", 657.2052001953125, 262.33331298828125], ["IfStmt", 715.7052612304688, 145.3333282470703],
            ["MethodCallExpr", 676.7052001953125, 184.33331298828125],
            ["getCredentialsPresent", 676.7052001953125, 223.33331298828125],
            ["BlockStmt", 754.7052612304688, 184.33331298828125],
            ["ExpressionStmt", 813.2052612304688, 145.3333282470703],
            ["VariableDeclarationExpr", 813.2052612304688, 184.33331298828125],
            ["ExpressionStmt", 891.2052001953125, 145.3333282470703],
            ["VariableDeclarationExpr", 891.2052001953125, 184.33331298828125],
            ["StatusLine", 852.2052612304688, 223.33331298828125],
            ["VariableDeclarator", 930.2052001953125, 223.33331298828125],
            ["statusLine", 891.2052001953125, 262.33331298828125],
            ["MethodCallExpr", 969.2052001953125, 262.33331298828125],
            ["ExpressionStmt", 978.9552001953125, 145.3333282470703],
            ["ExpressionStmt", 1066.7052001953125, 145.3333282470703],
            ["MethodCallExpr", 1066.7052001953125, 184.33331298828125],
            ["result", 988.7052001953125, 223.33331298828125],
            ["setStatusCode", 1066.7052001953125, 223.33331298828125],
            ["MethodCallExpr", 1144.7052001953125, 223.33331298828125],
            ["ExpressionStmt", 1242.2052001953125, 145.3333282470703],
            ["AssignExpr:assign", 1242.2052001953125, 184.33331298828125],
            ["FieldAccessExpr", 1203.2052001953125, 223.33331298828125],
            ["result", 1164.2052001953125, 262.33331298828125], ["strResponse", 1242.2052001953125, 262.33331298828125],
            ["MethodCallExpr", 1281.2052001953125, 223.33331298828125]]


def example_sort():
    return [
        # ["MethodDeclaration", 667.771728515625, 126.61770629882812], ["void", 515.896728515625, 156.61770629882812],
        #     ["f", 575.896728515625, 156.61770629882812], ["Parameter", 635.896728515625, 156.61770629882812],
        #     ["int", 605.896728515625, 186.6177215576172], ["array", 665.896728515625, 186.6177215576172],
        #     ["BlockStmt", 819.646728515625, 156.61770629882812],
        #     ["ExpressionStmt", 710.896728515625, 186.6177215576172],
        #     ["VariableDeclarationExpr", 710.896728515625, 216.6177215576172],
        #     ["boolean", 680.896728515625, 246.6177215576172],
        #     ["VariableDeclarator", 740.896728515625, 246.6177215576172],
        #     ["swapped", 710.896728515625, 276.61773681640625], ["true", 770.896728515625, 276.61773681640625],

            ["ForStmt", 928.396728515625, 186.6177215576172],
            ["VariableDeclarationExpr", 815.896728515625, 216.6177215576172],
            ["int", 785.896728515625, 246.6177215576172], ["VariableDeclarator", 845.896728515625, 246.6177215576172],
            ["i", 815.896728515625, 276.61773681640625], ["0", 875.896728515625, 276.61773681640625],
            ["BinaryExpr:and", 920.896728515625, 216.6177215576172],
            ["BinaryExpr:less", 890.896728515625, 246.6177215576172], ["swapped", 950.896728515625, 246.6177215576172],
            ["UnaryExpr:posIncrement", 980.896728515625, 216.6177215576172],
            ["BlockStmt", 1040.896728515625, 216.6177215576172],
            ["ExpressionStmt", 1010.896728515625, 246.6177215576172], ["ForStmt", 1070.896728515625, 246.6177215576172],
            ["VariableDeclarationExpr", 958.396728515625, 276.61773681640625],
            ["int", 928.396728515625, 306.61773681640625], ["VariableDeclarator", 988.396728515625, 306.61773681640625],
            ["j", 958.396728515625, 336.61773681640625], ["0", 1018.396728515625, 336.61773681640625],
            ["BinaryExpr:less", 1063.396728515625, 276.61773681640625], ["j", 1033.396728515625, 306.61773681640625],
            ["BinaryExpr:minus", 1093.396728515625, 306.61773681640625],
            ["BinaryExpr:minus", 1063.396728515625, 336.61773681640625], ["i", 1123.396728515625, 336.61773681640625],
            ["UnaryExpr:posIncrement", 1123.396728515625, 276.61773681640625],
            ["BlockStmt", 1183.396728515625, 276.61773681640625]]


# return original id(before @..)
def original_string(string):
    # if '@' in string:
    #     index = string.index('@')
    #     return string[:index]
    # return string
    return string.replace('@', '').replace(' ', '')


# l is a list extract from code2vec script
def create_graph(l):
    for i in range(len(l)):
        node = l[i][0]
        nodes_before = [x[0] for x in l[:i]]
        times_saw_string_before = len([x for x in nodes_before if original_string(x) == node])
        if times_saw_string_before:
            # l[i][0] += f'@{times_saw_string_before}'
            l[i][0] += f'@{" " * times_saw_string_before}'
    subtrees = []
    for i in range(len(l)):
        node, x, y = l[i]
        subtree = []
        for j in range(i + 1, len(l)):
            child, x_child, y_child = l[j]
            if y_child <= y:
                break
            subtree.append(l[j])
        subtrees.append((node, subtree))

    edges = []
    for node, subtree in subtrees:
        if len(subtree) > 0:
            child_y = min([x[-1] for x in subtree])
            childs = [x[0] for x in subtree if x[-1] == child_y]
            edges += [(node, child) for child in childs]

    g = nx.DiGraph()
    g.add_edges_from(edges)
    return WrapperGraph(g)


# l = copy_example()
# l = example_index_of()
l = example_sort()
# l = example_factorial()
# l = example_sum()
# l = example_http_post()
g = create_graph(l)
g.draw()

rules = [
    ('ExpressionStmt', 'VariableDeclarationExpr'),
    ('VariableDeclarationExpr', 'VariableDeclarator'),
    ('VariableDeclarator', '0'),
    ('VariableDeclarator', 'true'),
    ('VariableDeclarator', 'boolean'),
    ('VariableDeclarationExpr', 'int'),
    ('BlockStmt', 'ReturnStmt'),
    ('MethodDeclaration', 'BlockStmt'),
    ('BinaryExpr:equals', '0'),
    ('BinaryExpr:notEquals', 'null'),
    ('AssignExpr:assign', 'true'),
    ('ExpressionStmt', 'AssignExpr:assign'),
    ('ForStmt', 'VariableDeclarationExpr'),
    # ('ForStmt', 'BlockStmt'),
    # ('ForeachStmt', 'BlockStmt'),
    ('ForeachStmt', 'VariableDeclarationExpr'),
]

root = Node.from_graph(g)
original_size = root.size()
print(f'size before merges is {original_size}')
for parent_rule, child_rule in rules:
    print(f'merging {parent_rule, child_rule}')
    for node in root.all_nodes():
        for child in node.children:
            # simplifying by checking "in" rather than equlas
            if parent_rule in original_string(node.name) and child_rule in original_string(child.name):
                node.merge_nodes(child)
                break
    print(f'size after merges is {root.size()}')

print(f'removed {(1 - root.size() / original_size)*100}%  of nodes')

root.to_graph().draw()
