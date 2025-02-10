from typing import Union

class Value:
    "Each neuron is a value"
    def __init__(self, value, raw_value=None, op=None, children=[]):
        self.value = value
        self.raw_value = None
        if raw_value:
            self.raw_value = raw_value
        else:
            if isinstance(value, Value):
                self.raw_value = value.raw_value
            else:
                self.raw_value = value
        self.op = op
        self.children = children
        self.grads = []
        self.parents = []

    def __mul__(self, other) -> "Value":
        new_data = self.raw_value * other.raw_value if isinstance(other, Value) else self.raw_value * other
        new_data_val = Value(new_data)
        other = Value(other, other) if not isinstance(other, Value) else other
        new_value = Value(new_data_val, new_data, '*', children=[self, other])
        new_value.grads = [other, self]

        self.parents.append(new_value)
        other.parents.append(new_value)
        return new_value

    def __add__(self, other) -> "Value":
        new_data = self.raw_value + other.raw_value if isinstance(other, Value) else self.raw_value + other
        new_data_val = Value(new_data)
        other = Value(other, other) if not isinstance(other, Value) else other
        new_value = Value(new_data_val, new_data, '+', children=[self, other])
        new_value.grads = [1, 1]

        self.parents.append(new_value)
        other.parents.append(new_value)
        return new_value

class Neuron:
    "A neuron in a typical neural network"
    def __init__(self, value: Value):
        self.neuron_value = value
        self.neuron_rvalue = value.raw_value
        self.attrs = []
        self.grads = []
        self.cousins = {}
        self.__call__()

    #for multiply operation
    def __mul__(self, value: Union[Value, any]):
        if isinstance(value, Value):
            if value in self.attrs:
                value_idx = self.attrs.index(value)
                grad_value = self.grads[value_idx]
                new_grad_value = self.neuron_value + value * grad_value
                self.grads[value_idx] = new_grad_value
                self.neuron_value *= value.raw_value
                self.neuron_rvalue *= value.raw_value
                for cousin in self.cousins[value]:
                    cousin_idx = self.attrs.index(cousin)
                    cousin_grad = self.grads[cousin_idx]
                    new_grad_cousin = value * cousin_grad
                    self.grads[cousin_idx] = new_grad_cousin
        else:
            pass
    
    def __add__(self, value: Value) -> None:
        new_value = self.neuron_value + value
        self.neuron_value = new_value
        self.neuron_rvalue = new_value.raw_value
        children = self.neuron_value.children.copy()
        root_childrens = []
        while children:
            child = children.pop(0)
            if child.value == child.raw_value:
                root_childrens.append(child)
            if child.children != []:
                children.extend(child.children)
        for child in root_childrens:
            if child not in self.attrs:
                self.attrs.append(child)
                self.grads.append(0)

        del children

        return self.update_grad(transformation='+')
        # return None

    def update_grad(self, transformation=None) -> None:
        if transformation == None:
            raise Exception("update_grad was called without providing transformation")
        if transformation == '+':
            topos = []
            for attr in self.attrs:
                topo = self.till_neuron(attr)
                topos.append(topo)

            idx = 0
            while topos:
                topos_ = topos.pop(0)
                attr = self.attrs[idx]
                current_grad = 0
                for topo in topos_:
                    topo_grad = 1
                    topo.insert(0, attr)
                    for child, parent in zip(topo, topo[1:]):
                        grad_idx = parent.children.index(child)
                        grad_ = parent.grads[grad_idx]
                        grad = grad_.raw_value if isinstance(grad_, Value) else grad_
                        topo_grad *= grad
                    current_grad += topo_grad
                
                self.grads[idx] = current_grad
                idx += 1
        return topos

    def till_neuron(self, source: Value):
        topos = []
        def dfs(source, topo=[]):
            topo = topo
            if source == self.neuron_value:
                topos.append(topo.copy())
            parents = source.parents.copy()
            while parents:
                parent = parents.pop(0)
                topo.append(parent)
                dfs(parent, topo)
                topo = []
        dfs(source)
        return topos

    def __call__(self):
        self.attrs = [value for value in self.neuron_value.children]
        self.grads = self.neuron_value.grads.copy()
        for attr in self.attrs:
            attrs = self.attrs.copy()
            attrs.remove(attr)
            self.cousins[attr] = attrs


w = Value(2)
x = Value(3)
neuron = Neuron(w * x)
x_new = Value(5)
neuron + w * x_new
x_new_1 = Value(4)
neuron + w * x_new_1
neuron + w * x_new_1

print(neuron.grads)