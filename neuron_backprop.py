from typing import Union, Optional
import math

naked_attributes = []
their_vals = []

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
    
    def make_other_valid(self, other) -> "Value":
        if not isinstance(other, Value):
            if other not in naked_attributes:
                naked_attributes.append(other)
                other = Value(other)      
                their_vals.append(other)
            else:
                other = their_vals[naked_attributes.index(other)]
        return other

    def __mul__(self, other) -> "Value":
        new_data = self.raw_value * other.raw_value if isinstance(other, Value) else self.raw_value * other
        new_data_val = Value(new_data)
        other = self.make_other_valid(other)
        new_value = Value(new_data_val, new_data, '*', children=[self, other])
        new_value.grads = [other, self]

        self.parents.append(new_value)
        other.parents.append(new_value)
        return new_value
   
    def __rmul__(self, other):
        return self * other

    def __add__(self, other) -> "Value":
        new_data = self.raw_value + other.raw_value if isinstance(other, Value) else self.raw_value + other
        new_data_val = Value(new_data)
        other = self.make_other_valid(other)
        new_value = Value(new_data_val, new_data, '+', children=[self, other])
        new_value.grads = [1, 1]

        self.parents.append(new_value)
        other.parents.append(new_value)
        return new_value

    def __radd__(self, other):
        return self + other
    
    def log(self):
        if self.raw_value <= 0:
            raise Exception(f"log of negative value: {self.raw_value}, is undefined. At least I cannot calculate it.")
        else:
            new_data = math.log(self.raw_value)
            new_data_val = Value(new_data, )
            new_value = Value(new_data_val, new_data, '*', children=[self])
            new_value.grads = [1/self.raw_value]
            self.parents.append(new_value)
            return new_value


class Neuron:
    "A neuron in a typical neural network"
    def __init__(self, value: Optional[Value]=None, next=False, prev_neuron: Optional["Neuron"]=None):
        if isinstance(value, Value):
            self.neuron_value = value
            self.neuron_rvalue = value.raw_value
        self.attrs = []
        self.grads = []
        self.cousins = {}
        if not next:
            self.__call__()
        else:
            if prev_neuron != None:
                self.prev_neuron = prev_neuron
            else:
                raise Exception("Cannot go to next neuron without a previous neuron.")

    #for multiply operation
    def __mul__(self, value: Union[Value, any]) -> "Neuron":
        new_value = self.neuron_value * value
        self.neuron_value = new_value
        self.neuron_rvalue = new_value.raw_value
        self.get_roots_and_update_attrs_()
        self.update_grad(transformation='*')
        return self
    
    def __add__(self, value: Union[Value, any]) -> "Neuron":
        new_value = self.neuron_value + value
        self.neuron_value = new_value
        self.neuron_rvalue = new_value.raw_value
        self.get_roots_and_update_attrs_()
        self.update_grad(transformation='+')
        return self
        # return None

    def get_roots_and_update_attrs_(self) -> None:
        children = self.neuron_value.children.copy()
        root_children = []
        while children:
            child = children.pop(0)
            if child.value == child.raw_value:
                root_children.append(child)
            if child.children != []:
                children.extend(child.children)
        for child in root_children:
            if child not in self.attrs:
                self.attrs.append(child)
                self.grads.append(0)

        del children, root_children
        return None

    def update_grad(self, transformation=None) -> None:
        if transformation == None:
            raise Exception("update_grad was called without providing transformation")
        else:
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
    
    def neuron_next_step(self, op='next', idxd_nnw=None):
        #creates a neuron in the next step
        if op == "next":
            if idxd_nnw != None:
                new_neuron = Neuron(next=True, prev_neuron=self)
                new_neuron.neuron_value = idxd_nnw * self.neuron_value
                new_neuron.neuron_rvalue = new_neuron.neuron_value.raw_value
                new_neuron()
                for attr in new_neuron.attrs:
                    print(attr.raw_value)

    def __call__(self):
        self.attrs = [value for value in self.neuron_value.children]
        self.grads = self.neuron_value.grads.copy()
        for attr in self.attrs:
            attrs = self.attrs.copy()
            attrs.remove(attr)
            self.cousins[attr] = attrs
