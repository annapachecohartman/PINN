import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
from torch import pi, sin, cos
from sklearn.model_selection import train_test_split
from sciann_datagenerator import *

# Pytorch implementation of SciANN implementation of a forward Linear Elasticity problem

# Values of lambda and mu are given
# lmbd = 1.0
# mu = 0.5
# Q = 4.0

# data generator plot example
# dg = DataGeneratorXY([0.,1.], [0.,1.], num_sample=1000)
# dg.plot_data()


# Functional class from SciANN implementation in pytorch
# Creates a neural network with specified layers and nodes
# with default activation function as tanh
# and input size (in this case input_size = 2)
class Functional(nn.Module):
    def __init__(self,input_size, layer_sizes, activation='tanh'):
        super(Functional, self).__init__()

        if activation == 'tanh':
            self.activation = torch.tanh

        self.hidden_layer1 = nn.Linear(input_size, layer_sizes[0])
        self.final_layer = nn.Linear(layer_sizes[-1], 1)

        self.n_layers= len(layer_sizes)
        # list of layers with the corresponding nodes in layer_sizes list
        self.layers_list = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.n_layers - 1)])

    def forward(self, inputs):
        layer_out = self.activation(self.hidden_layer1(inputs))
        for layer in range(self.n_layers - 1):
            layer_out = self.activation(self.layers_list[layer](layer_out))
        layer_output = torch.squeeze(self.activation(self.final_layer(layer_out)))  # final output from final layer

        return layer_output


# Implementation of Tie class (which is a part of the Constraint class in SciANN)
# constrains the cond1 and cond2 given to follow: cond1 - cond2 == sol
class Tie:
    # cond1 and cond2 are functional objects
    # constraint: `cond1 - cond2 == sol`.
    def __init__(self, cond1, cond2):
        try:
            self.cond = cond1 - cond2
        except (ValueError, TypeError):
            print(
                'Unexpected ValueError/TypeError - ',
                'make sure `cond1` and `cond2` are functional objects. \n',
                'cond1 - {} \n'.format(cond1),
                'cond2 - {} \n'.format(cond2)
            )

    def __call__(self):
        return self.cond

    def eval(self, xs):
        return self.cond.eval(xs)

def pde_function(x, y, Uxy, Vxy):
    lmbd = 1.0
    mu = 0.5
    Q = 4.0
    # Both forces below as written as the negatives of the forces written in the paper,
    # as they are used for the loss function
    # which is written from the momentum balance equation
    # Fx represents the fx body force as written in the paper
    Fx = - lmbd * (4 * pi ** 2 * cos(2 * pi * x) * sin(pi * y) - Q * y ** 3 * pi * cos(pi * x)) \
         - mu * (pi ** 2 * cos(2 * pi * x) * sin(pi * y) - Q * y ** 3 * pi * cos(pi * x)) \
         - 8 * mu * pi ** 2 * cos(2 * pi * x) * sin(pi * y)
    # Fy represents the fy body force as written in the paper
    Fy = lmbd * (3 * Q * y ** 2 * sin(pi * x) - 2 * pi ** 2 * cos(pi * y) * sin(2 * pi * x)) \
         - mu * (2 * pi ** 2 * cos(pi * y) * sin(2 * pi * x) + (Q * y ** 4 * pi ** 2 * sin(pi * x)) / 4) \
         + 6 * Q * mu * y ** 2 * sin(pi * x)

    # Constants used in the equations
    C11 = 2 * mu + lmbd  # λ + 2µ
    C12 = lmbd  # λ
    C33 = 2 * mu  # 2µ

    grads_Uxy_x = torch.autograd.grad(Uxy, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    grads_Vxy_y = torch.autograd.grad(Vxy, y, torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    grads_Uxy_y = torch.autograd.grad(Uxy, y, torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    grads_Vxy_x = torch.autograd.grad(Vxy, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]

    # Epsilon represents infinitesimal stress tensor, which can be found
    # by differentiating the displacement (according to the kinematic relations equation)
    Exx = grads_Uxy_x
    Eyy = grads_Vxy_y
    Exy = grads_Uxy_y * grads_Vxy_x * 0.5

    Sxx = Exx * C11 + Eyy * C12  # σxx = (λ + 2µ)εxx + λ * εyy
    Syy = Eyy * C11 + Exx * C12  # σyy = (λ + 2µ)εyy + λ * εxx
    Sxy = Exy * C33  # σxy = 2µ * εxy

    # The constants represent the constraints from the constitutive model equation
    # c1 = Tie(Sxx, Exx * C11 + Eyy * C12) # σxx = (λ + 2µ)εxx + λεyy
    # c2 = Tie(Syy, Eyy * C11 + Exx * C12) # σyy = (λ + 2µ)εyy + λεxx
    # c3 = Tie(Sxy, Exy * C33) # σxy = 2µεxy

    grads_Sxx_x = torch.autograd.grad(Sxx, x, torch.ones(x.shape), create_graph=True, retain_graph=True)[0]
    grads_Sxy_y = torch.autograd.grad(Sxy, y, torch.ones(y.shape), create_graph=True, retain_graph=True)[0]
    grads_Sxy_x = torch.autograd.grad(Sxy, x, torch.ones(x.shape), create_graph=True, retain_graph=True)[0]
    grads_Syy_y = torch.autograd.grad(Syy, y, torch.ones(y.shape), create_graph=True, retain_graph=True)[0]

    # Loss calculated as the difference between the values of sigma outputted and the given body forces
    # as described by the momentum balance equation: σij,j = -fi (rearranged from σij,j + fi = 0)
    Lx = grads_Sxx_x + grads_Sxy_y - Fx
    Ly = grads_Sxy_x + grads_Syy_y - Fy

    # Boundary conditions

    BC_left_1 = (x == 0.) * Vxy  # when x is 0, Vxy (meaning Uy) should equal 0 as well.
    BC_left_2 = (x == 0.) * Sxx  # when x is 0, σxx should equal 0 as well.

    BC_right_1 = (x == 1.) * Vxy  # when x is 1, Vxy (meaning Uy) should equal 0.
    BC_right_2 = (x == 1.) * Sxx  # when x is 1, σxx should equal 0.

    BC_bot_1 = (y == 0.) * Uxy  # when y is 0, Uxy (meaning Ux) should equal 0.
    BC_bot_2 = (y == 0.) * Vxy  # when y is 0, Vxy (meaning Uy) should equal 0.

    BC_top_1 = (y == 1.) * Uxy  # when y is 0, Uxy (meaning Ux) should equal 0.
    BC_top_2 = (y == 1.) * (Syy - (lmbd + 2 * mu) * Q * sin(pi * x))  # when y is 0, σyy = (λ + 2µ)Q
    # * sin(πx)
    # ^this can be rewritten as σyy - ((λ + 2µ)Q * sin(πx)) = 0

    return Lx, Ly, Sxx, Syy, Sxy, BC_left_1, BC_left_2, BC_right_1, BC_right_2, BC_bot_1, BC_bot_2, BC_top_1, BC_top_2



# PINN class contains the main implementation of the SciANN training model
# with loss function and definition of variables/tensors
class PINN(nn.Module):
    def __init__(self, input_size, Uxy_sizes, Vxy_sizes, Sxx_sizes, Syy_sizes, Sxy_sizes):
        super(PINN, self).__init__()
        self.Uxy_fn = Functional(input_size,Uxy_sizes)
        self.Vxy_fn = Functional(input_size, Vxy_sizes)

        self.Sxx_fn = Functional(input_size, Sxx_sizes)
        self.Syy_fn = Functional(input_size, Syy_sizes)
        self.Sxy_fn = Functional(input_size, Sxy_sizes)


        self.mu = 1.0
        self.lmbd = 0.5
        self.Q = 4.0


    def forward(self, x, y):
        xy_data = torch.stack([x, y], dim=1)


        # Outputs from the FFNNs defined as Functional objects given the x and y inputs
        Uxy = self.Uxy_fn(xy_data)  # Ux
        Vxy = self.Vxy_fn(xy_data)  # Uy



        Lx, Ly, Sxx, Syy, Sxy, BC_left_1, BC_left_2, BC_right_1, BC_right_2, BC_bot_1, BC_bot_2, BC_top_1, BC_top_2 = pde_function(x, y, Uxy, Vxy)

        return Lx, Ly, Sxx, Syy, Sxy, BC_left_1, BC_left_2, BC_right_1, BC_right_2, BC_bot_1, BC_bot_2, BC_top_1, BC_top_2

    def plot(self, x, y):
        xy_data = torch.stack([x, y], dim=1)

        # Uxy = torch.tensor(self.Uxy_fn(xy_data), requires_grad=True)
        # Vxy = torch.tensor(self.Vxy_fn(xy_data), requires_grad=True)
        Uxy = self.Uxy_fn(xy_data)
        Vxy = self.Vxy_fn(xy_data)

        # Sxx = self.Sxx_fn(xy_data)  # σxx
        # Syy = self.Syy_fn(xy_data)  # σyy
        # Sxy = self.Sxy_fn(xy_data)  # σxy
        Lx, Ly, Sxx, Syy, Sxy, BC_left_1, BC_left_2, BC_right_1, BC_right_2, BC_bot_1, BC_bot_2, BC_top_1, BC_top_2 = pde_function(
            x, y, Uxy, Vxy)

        return Uxy, Vxy, Sxx, Syy, Sxy

# Data from data generator
dg = DataGeneratorXY(X=[0.,1.], Y=[0.,1.],
                     num_sample=10000,
                     targets=5*['domain'] + 2*['bc-left'] + 2*['bc-right'] + 2*['bc-bot'] + 2*['bc-top'])

input_data, target_data = dg.get_data()

x_data = input_data[0] # 12500 points
y_data = input_data[1]

x_data = x_data.reshape((-1))
y_data = y_data.reshape((-1))
print(x_data.shape)

# Lx_data = target_data[0][1]
# print(Lx_data)
# Ly_data = target_data[1]
# c1_data = target_data[2]
#
# c2_data = target_data[3]
# c3_data = target_data[4]

BC_left1_data = target_data[5]
BC_left2_data = target_data[6]
BC_right1_data = target_data[7]
BC_right2_data = target_data[8]
BC_bot1_data = target_data[9]
BC_bot2_data = target_data[10]
BC_top1_data = target_data[11]
BC_top2_data = target_data[12]

# print(len(target_data))
# print(target_data[5][0])
# input_train, input_test, target_train, target_test = train_test_split(input_data, target_data)

# Training
model = PINN(2, 4*[30], 4*[30], 4*[30], 4*[30], 4*[30])
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) # 0.001 or 0.01 ?
mse = torch.nn.MSELoss(reduction='mean')
# mse = mse.cuda()
n_epochs = 5000
batch_size = 200
avg_train_loss = []
avg_bc_loss = []
avg_LXY_loss = []
# avg_c_loss = []


for epoch in range(n_epochs):
    train_loss_list = []
    bc_loss_list = []
    LXY_loss_list = []
    # c_loss_list = []
    n_batches = np.ceil(x_data.shape[0]/batch_size).astype(int) # number of batches
    model.train()
    for batch_i in range(n_batches):
        batch_start = batch_i * batch_size
        batch_stop = np.minimum(x_data.shape[0], (batch_i + 1) * batch_size)

        x_batch = torch.from_numpy(x_data[batch_start:batch_stop]).to(dtype=torch.float32)
        y_batch = torch.from_numpy(y_data[batch_start:batch_stop]).to(dtype=torch.float32)
        # x_batch = input_batch[0]
        # print(x_batch)
        # y_batch = input_batch[1]
        # print()
        # print(y_batch)
        # Lx_batch = torch.from_numpy(Lx_data[batch_start:batch_stop]).to(dtype=torch.float32)
        # Ly_batch = torch.from_numpy(Ly_data[batch_start:batch_stop]).to(dtype=torch.float32)

        # c1_batch = torch.from_numpy(c1_data[batch_start:batch_stop]).to(dtype=torch.float32)
        # c2_batch = torch.from_numpy(c2_data[batch_start:batch_stop]).to(dtype=torch.float32)
        # c3_batch = torch.from_numpy(c3_data[batch_start:batch_stop]).to(dtype=torch.float32)

        # how to find mean square error of boundary conditions??
        BCleft1_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        BCleft2_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        BCright1_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        BCright2_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        BCbot1_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        BCbot2_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        BCtop1_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        BCtop2_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)

        Lx_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        Ly_batch = torch.from_numpy(np.zeros_like(y_batch)).to(dtype=torch.float32)
        # c1_batch = torch.from_numpy(np.zeros_like(y_batch)).to(dtype=torch.float32)
        # c2_batch = torch.from_numpy(np.zeros_like(y_batch)).to(dtype=torch.float32)
        # c3_batch = torch.from_numpy(np.zeros_like(y_batch)).to(dtype=torch.float32)

        x_batch.requires_grad = True
        y_batch.requires_grad = True

        optimizer.zero_grad()
        Lx_hat, Ly_hat, Sxx_hat, Syy_hat, Sxy_hat, BC_left1_hat, BC_left2_hat, BC_right1_hat, BC_right2_hat,\
            BC_bot1_hat, BC_bot2_hat, BC_top1_hat, BC_top2_hat = model(x_batch, y_batch)

        Lx_loss = mse(Lx_hat, Lx_batch)
        Ly_loss = mse(Ly_hat, Ly_batch)
        # c1_loss = mse(c1_hat, c1_batch)
        # c2_loss = mse(c2_hat, c2_batch)
        # c3_loss = mse(c3_hat, c3_batch)

        BCleft1_loss = mse(BC_left1_hat, BCleft1_batch)
        BCleft2_loss = mse(BC_left2_hat, BCleft2_batch)
        BCright1_loss = mse(BC_right1_hat, BCright1_batch)
        BCright2_loss = mse(BC_right2_hat, BCright2_batch)
        BCbot1_loss = mse(BC_bot1_hat, BCbot1_batch)
        BCbot2_loss = mse(BC_bot2_hat, BCbot2_batch)
        BCtop1_loss = mse(BC_top1_hat, BCtop1_batch)
        BCtop2_loss = mse(BC_top2_hat, BCtop2_batch)

        total_LXY_loss = Lx_loss + Ly_loss
        # total_c_loss = c1_loss + c2_loss + c3_loss
        # print(f'BC loss {BCtop1_loss.item() + BCtop2_loss.item() + BCbot2_loss.item() + BCbot1_loss.item() +  BCleft1_loss.item() + BCleft2_loss.item() + BCright1_loss.item() + BCright2_loss.item()}')
        # print(f'CM eq loss {c1_loss.item() + c2_loss.item() + c3_loss.item()}')
        # print(f'LxLy loss {Lx_loss.item() + Ly_loss.item()}')
        total_bc_loss = BCleft1_loss + BCleft2_loss + BCright1_loss + BCright2_loss + BCbot1_loss + BCbot2_loss + \
            BCtop1_loss + BCtop2_loss
        total_loss = total_LXY_loss + total_bc_loss
        total_loss.backward()
        optimizer.step()
        train_loss_list.append(total_loss.item())
        bc_loss_list.append(total_bc_loss.item())
        # c_loss_list.append(total_c_loss.item())
        LXY_loss_list.append(total_LXY_loss.item())
    avg_train_loss.append(np.mean(train_loss_list))
    avg_bc_loss.append(np.mean(bc_loss_list))
    # avg_c_loss.append(np.mean(c_loss_list))
    avg_LXY_loss.append(np.mean(LXY_loss_list))

    # Validate
    # model.eval()
    # val_loss_list = []
    # n_batches_val = np.ceil(input_test.shape[0] / batch_size).astype(int)
    #
    # for batch_i in range(n_batches):
    #     batch_start = batch_i * batch_size
    #     batch_stop = np.minimum(input_test.shape[0], (batch_i + 1) * batch_size)
    #
    #     x_batch = torch.from_numpy(input_test[batch_start:batch_stop]).to(dtype=torch.float32)
    #     u_batch = torch.from_numpy(target_test[batch_start:batch_stop]).to(dtype=torch.float32)
    #     Lx_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
    #     Ly_batch = torch.from_numpy(np.zeros_like(y_batch)).to(dtype=torch.float32)
    #
    #     x_batch.requires_grad = True
    #     y_batch.requires_grad = True
    #
    #     Lx_hat, Ly_hat, Uxy_hat, Vxy_hat = model(x_batch, y_batch)
    #
    #     Lx_loss = mse(Lx_hat, Lx_batch)
    #     Ly_loss = mse(Ly_hat, Ly_batch)
    #     Uxy_loss = mse(Uxy_hat, u_batch)
    #     Vxy_loss = mse(Vxy_hat, v_batch)
    #
    #     total_loss = Lx_loss + Ly_loss + Uxy_loss + Vxy_loss
    #     val_loss_list.append(total_loss.item())

    if epoch % 50 == 0:
        print("Epoch " + str(epoch) + ": Training total loss: " + str(avg_train_loss[epoch]) + ", BC loss: " + str(avg_bc_loss[epoch]) + ", LXY loss: " + str(avg_LXY_loss[epoch]))
        # print("Epoch " + str(epoch) + ": Validation loss: " + str(val_loss_list[epoch]))
# Test

x_plot, y_plot = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
x_plot2, y_plot2 = np.meshgrid(np.linspace(0,1,100), np.linspace(0,1,100))
x_plot2 = x_plot2.reshape((-1))
y_plot2 = y_plot2.reshape((-1))
x_plot2 = torch.from_numpy(x_plot2).to(dtype=torch.float32)
y_plot2 = torch.from_numpy(y_plot2).to(dtype=torch.float32)
x_plot2.requires_grad=True
y_plot2.requires_grad=True

Uxy_p, Vxy_p, Sxx_p, Syy_p, Sxy_p = model.plot(x_plot2, y_plot2)
Uxy_p = Uxy_p.reshape((100, 100))
Vxy_p = Vxy_p.reshape((100, 100))
Sxx_p = Sxx_p.reshape((100, 100))
Syy_p = Syy_p.reshape((100, 100))
Sxy_p = Sxy_p.reshape((100, 100))
plot_dict = {
    'u':Uxy_p.detach().numpy(),
    'v':Vxy_p.detach().numpy(),
    'sxx':Sxx_p.detach().numpy(),
    'syy':Syy_p.detach().numpy(),
    'sxy':Sxy_p.detach().numpy()
}

plt.plot(avg_train_loss, label='train_loss')
plt.legend()
plt.show()

def cust_plot(ax, val, label):
    im = ax.pcolor(x_plot, y_plot, val, cmap='seismic', shading='auto',
                   vmin=-np.abs(val).max(), vmax=np.abs(val).max())
    ax.set_title(label)
    plt.colorbar(im, ax=ax)


fig, ax = plt.subplots(1, 5, figsize=(15, 3))
for i, (key, val) in enumerate(plot_dict.items()):
    cust_plot(ax[i], val, key)

plt.tight_layout()
plt.show()
