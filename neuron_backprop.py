from typing import Union

class Value:
    "Each neuron is a value"
    def __init__(self, value, raw_value=None, op=None, children=[]):
        self.value = value
        self.raw_value = raw_value if raw_value else value 
        self.op = op
        self.children = children
        self.grads = []

    def __mul__(self, other):
        new_data = self.raw_value * other.raw_value if isinstance(other, Value) else self.raw_value * other
        other_value = Value(other, other) if not isinstance(other, Value) else other
        new_value = Value(new_data, new_data, '*', children=[self, other_value])
        new_value.grads = [other_value, self]
        return new_value

    def __add__(self, other):
        new_data = self.raw_value + other.raw_value if isinstance(other, Value) else self.raw_value + other
        other_value = Value(other, other) if not isinstance(other, Value) else other
        new_value = Value(new_data, new_data, '+', children=[self, other_value])
        new_value.grads = [1, 1]
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

    def __call__(self):
        self.attrs = [value for value in self.neuron_value.children]
        self.grads = self.neuron_value.grads.copy()
        for attr in self.attrs:
            attrs = self.attrs.copy()
            attrs.remove(attr)
            self.cousins[attr] = attrs

value = Value(2)
new_value = value * 1
neuron = Neuron(new_value)
neuron * value
for grad_val in neuron.grads:
    print(grad_val.value)