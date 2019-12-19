'''
0.0
(1-tanh^2(tanh(tanh(tanh(100))))) * (1-tanh^2(tanh(tanh(100)))) * (1-tanh^2(tanh(100))) * (1-tanh^2(100)) * 100
0.168
(1-tanh^2(tanh(tanh(tanh(100))))) * (1-tanh^2(tanh(tanh(100)))) * (1-tanh^2(tanh(100))) * tanh(100)
0.304
(1-tanh^2(tanh(tanh(tanh(100))))) * (1-tanh^2(tanh(tanh(100)))) * tanh(tanh(100))
0.436
(1-tanh^2(tanh(tanh(tanh(100))))) * tanh(tanh(tanh(100)))

100.0
ReLU'(ReLU(ReLU(ReLU(100)))) * ReLU'(ReLU(ReLU(100))) * ReLU'(ReLU(100)) * ReLU'(100) * 100
100.0
ReLU'(ReLU(ReLU(ReLU(100)))) * ReLU'(ReLU(ReLU(100))) * ReLU'(ReLU(100)) * ReLU(100)
100.0
ReLU'(ReLU(ReLU(ReLU(100)))) * ReLU'(ReLU(ReLU(100))) * ReLU(ReLU(100))
100.0
ReLU'(ReLU(ReLU(ReLU(100)))) * ReLU(ReLU(ReLU(100)))

[0.0, 0.168, 0.304, 0.436], [100.0, 100.0, 100.0, 100.0]
'''
