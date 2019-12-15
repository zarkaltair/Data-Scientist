import torch

N = 4
C = 3
C_out = 10
H = 8
W = 16

x = torch.ones((N, C, H, W))
out = torch.nn.Conv2d(C, C_out, kernel_size=(3, 3), padding=1)(x)
print(out.shape)  # torch.Size([4, 10, 8, 16])

out = torch.nn.Conv2d(C, C_out, kernel_size=(5, 5), padding=2)(x)
print(out.shape)  # torch.Size([4, 10, 8, 16])

out = torch.nn.Conv2d(C, C_out, kernel_size=(7, 7), padding=3)(x)
print(out.shape)  # torch.Size([4, 10, 8, 16])

out = torch.nn.Conv2d(C, C_out, kernel_size=(9, 9), padding=4)(x)
print(out.shape)  # torch.Size([4, 10, 8, 16])

out = torch.nn.Conv2d(C, C_out, kernel_size=(3, 5), padding=(1, 2))(x)
print(out.shape)  # torch.Size([4, 10, 8, 16])

out = torch.nn.Conv2d(C, C_out, kernel_size=(3, 3), padding=8)(x)
print(out.shape)  # torch.Size([4, 10, 22, 30])

out = torch.nn.Conv2d(C, C_out, kernel_size=(4, 4), padding=1)(x)
print(out.shape)  # torch.Size([4, 10, 7, 15])

out = torch.nn.Conv2d(C, C_out, kernel_size=(2, 2), padding=1)(x)
print(out.shape)  # torch.Size([4, 10, 9, 17])