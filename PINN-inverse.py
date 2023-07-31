import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
from torch import pi, sin, cos
# from functorch import grad
from sklearn.model_selection import train_test_split


# Pytorch implementation of SciANN implementation of inverse Linear Elasticity problem: goal is to find mu and lambda
Q = 4.0 # constant used in the equations

class Functional(nn.Module):
    def __init__(self,input_size, layer_sizes, activation='tanh'):
        super(Functional, self).__init__()

        if activation == 'tanh': # activation function is tanh
            self.activation = torch.tanh

        # first layer is 2 (input_size) nodes & num of nodes in the layer_sizes[0]
        self.hidden_layer1 = nn.Linear(input_size, layer_sizes[0])
        self.final_layer = nn.Linear(layer_sizes[-1], 1) # final layer is nodes in layer_sizes[-1] (last item) and 1

        self.n_layers= len(layer_sizes) # number of layers (not counting input and output layers)
        # list of layers with the corresponding nodes in layer_sizes list
        self.layers_list = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.n_layers - 1)])

    def forward(self, inputs):
        layer_out = self.activation(self.hidden_layer1(inputs)) # output of first layer
        # loop through the layers in layers_list
        for layer in range(self.n_layers - 1):
            layer_out = self.activation(self.layers_list[layer](layer_out))

        layer_output = torch.squeeze(self.activation(self.final_layer(layer_out))) # final output from final layer

        return layer_output


def pde_function(x, y, Uxy, Vxy, mu, lmbd, plot=False):
    # Both forces below as written as the negatives of the forces written in the paper,
    #         # as they are used for the loss function
    #         # which is written from the momentum balance equation
    #         # Fx represents the fx body force as written in the paper
    Fx = - 1. * (4 * pi ** 2 * cos(2 * pi * x) * sin(pi * y) - Q * y ** 3 * pi * cos(pi * x))\
         - 0.5 * (pi ** 2 * cos(2 * pi * x) * sin(pi * y) - Q * y ** 3 * pi * cos(pi * x)) \
         - 0.5 * 8 * pi ** 2 * cos(2 * pi * x) * sin(pi * y)

    # Fy represents the fy body force as written in the paper
    Fy = 1.0 * (3 * Q * y ** 2 * sin(pi * x) - 2 * pi ** 2 * cos(pi * y) * sin(2 * pi * x)) \
         - 0.5 * (2 * pi ** 2 * cos(pi * y) * sin(2 * pi * x) + (Q * y ** 4 * pi ** 2 * sin(pi * x)) / 4) \
         + 0.5 * 6 * Q * y ** 2 * sin(pi * x)

    # Constants used in the equations
    C11 = 2 * mu + lmbd  # λ + 2µ
    C12 = lmbd  # λ
    C33 = 2 * mu  # 2µ

    # Grads for E
    grads_Uxy_x = torch.autograd.grad(Uxy, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    grads_Vxy_y = torch.autograd.grad(Vxy, y, torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    grads_Uxy_y = torch.autograd.grad(Uxy, y, torch.ones_like(y), create_graph=True, retain_graph=True)[0]
    grads_Vxy_x = torch.autograd.grad(Vxy, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]

    # Epsilon represents infinitesimal stress tensor, which can be found
    # by differentiating the displacement (according to the kinematic relations equation)
    Exx = grads_Uxy_x
    Eyy = grads_Vxy_y
    Exy = grads_Uxy_y + grads_Vxy_x * 0.5

    # Sigma is calculated using the constitutive model equation
    Sxx = Exx * C11 + Eyy * C12 # σxx = (λ + 2µ)εxx + λ * εyy
    Syy = Eyy * C11 + Exx * C12 # σyy = (λ + 2µ)εyy + λ * εxx
    Sxy = Exy * C33 # σxy = 2µ * εxy

    # Grads for L
    grads_Sxx_x = torch.autograd.grad(Sxx, x, torch.ones(x.shape), create_graph=True, retain_graph=True)[0]
    grads_Sxy_y = torch.autograd.grad(Sxy, y, torch.ones(y.shape), create_graph=True, retain_graph=True)[0]
    grads_Sxy_x = torch.autograd.grad(Sxy, x, torch.ones(x.shape), create_graph=True, retain_graph=True)[0]
    grads_Syy_y = torch.autograd.grad(Syy, y, torch.ones(y.shape), create_graph=True, retain_graph=True)[0]

    # Loss calculated as the difference between the values of sigma outputted and the given body forces
    # as described by the momentum balance equation: σij,j = -fi (rearranged from σij,j + fi = 0)
    Lx = grads_Sxx_x + grads_Sxy_y - Fx
    Ly = grads_Sxy_x + grads_Syy_y - Fy

    if plot:
        return Sxx, Syy, Sxy
    else:
        return Lx, Ly

class PINN(nn.Module):
    def __init__(self, input_size, Uxy_sizes, Vxy_sizes):
        super(PINN, self).__init__()
        self.Uxy_fn = Functional(input_size,Uxy_sizes)  # Ux as outputted by the Functional NN
        self.Vxy_fn = Functional(input_size, Vxy_sizes)  # Ux as outputted by the Functional NN

        self.mu = torch.nn.parameter.Parameter(data=torch.randn(1) + 1, requires_grad=True)  # mu is a goal parameter
        self.lmbd = torch.nn.parameter.Parameter(data=torch.randn(1) + 1, requires_grad=True)  # lambda is a goal parameter

    def forward(self, x, y):
        # x_ = x.clone()
        # x_.requires_grad = True
        # y_ = y.clone()
        # y_.requires_grad = True
        xy_data = torch.stack([x, y], dim=1)

        # Uxy = torch.tensor(self.Uxy_fn(xy_data), requires_grad=True)
        # Vxy = torch.tensor(self.Vxy_fn(xy_data), requires_grad=True)
        Uxy = self.Uxy_fn(xy_data)
        Vxy = self.Vxy_fn(xy_data)

        Lx, Ly = pde_function(x, y, Uxy, Vxy, self.mu, self.lmbd)
        return Lx, Ly, Uxy, Vxy

    def test(self, x, y, Uxy, Vxy):
        Lx, Ly = pde_function(x, y, Uxy, Vxy, self.mu, self.lmbd)
        return Lx, Ly

    def plot(self, x, y):
        xy_data = torch.stack([x, y], dim=1)

        # Uxy = torch.tensor(self.Uxy_fn(xy_data), requires_grad=True)
        # Vxy = torch.tensor(self.Vxy_fn(xy_data), requires_grad=True)
        Uxy = self.Uxy_fn(xy_data)
        Vxy = self.Vxy_fn(xy_data)

        Sxx, Syy, Sxy = pde_function(x, y, Uxy, Vxy, self.mu, self.lmbd, True)
        return Uxy, Vxy, Sxx, Syy, Sxy





# class PINN(nn.Module):
#     def __init__(self, input_size, Uxy_sizes, Vxy_sizes):
#         super(PINN, self).__init__()
#         self.Uxy_fn = Functional(input_size,Uxy_sizes)
#         self.Vxy_fn = Functional(input_size, Vxy_sizes)
#
#         self.mu = torch.nn.parameter.Parameter(requires_grad=True)
#         self.lmbd = torch.nn.parameter.Parameter(requires_grad=True)
#
#     def forward(self, x, y):
#
#         xy_data = torch.stack([x, y], dim=1)
#         # print(xy_data.shape)
#         # Both forces below as written as the negatives of the forces written in the paper,
#         # as they are used for the loss function
#         # which is written from the momentum balance equation
#         # Fx represents the fx body force as written in the paper
#         # print(x.shape)
#         # print(x.dtype)
#         # print(y.shape)
#         # print(y.dtype)
#         Fx = - 1. * (4 * pi ** 2 * cos(2 * pi * x) * sin(pi * y) - Q * y ** 3 * pi * cos(pi * x)) - 0.5 * (pi ** 2 * cos(2 * pi * x) * sin(pi * y) - Q * y ** 3 * pi * cos(pi * x)) - 0.5 * 8 * pi ** 2 * cos(2 * pi * x) * sin(pi * y)
#
#         # Fy represents the fy body force as written in the paper
#         Fy = 1.0 * (3 * Q * y ** 2 * sin(pi * x) - 2 * pi ** 2 * cos(pi * y) * sin(2 * pi * x)) \
#              - 0.5 * (2 * pi ** 2 * cos(pi * y) * sin(2 * pi * x) + (Q * y ** 4 * pi ** 2 * sin(pi * x)) / 4) \
#              + 0.5 * 6 * Q * y ** 2 * sin(pi * x)
#
#         # Constants used in the equations
#         C11 = 2 * self.mu + self.lmbd  # λ + 2µ
#         C12 = self.lmbd  # λ
#         C33 = 2 * self.mu  # 2µ
#
#
#         # Outputs from the FFNNs defined as Functional objects given the x and y inputs
#         Uxy = self.Uxy_fn(xy_data).clone().detach().requires_grad_(True) # Ux torch.tensor?? .sum()?
#         Vxy = self.Vxy_fn(xy_data).clone().detach().requires_grad_(True) # Uy torch.tensor?? .sum()?
#
#
#         # Uxy.backward()
#         # Vxy.backward()
#
#         # Epsilon represents infinitesimal stress tensor, which can be found
#         # by differentiating the displacement (according to the kinematic relations equation)
#         Exx_list = []
#         Eyy_list = []
#         Exy_list = []
#
#         # print(torch.autograd.grad(Uxy, x, is_grads_batched=True))
#         print(torch.autograd.grad(Vxy[0], y[0], allow_unused=True))
#         print(torch.autograd.grad(Uxy[0], y[0], allow_unused=True))
#         print(torch.autograd.grad(Vxy[0], x[0], allow_unused=True))
#
#         for i in range(x.shape[0]):
#             Exx_list.append(torch.autograd.grad(Uxy[i], x[i], allow_unused=True)[0])
#             Eyy_list.append(torch.autograd.grad(Vxy[i], y[i], allow_unused=True)[0])
#             Exy_list.append(torch.autograd.grad(Uxy[i], y[i], allow_unused=True)[0] + torch.autograd.grad(Vxy[i], x[i], allow_unused=True)[0] * 0.5)
#         # Exx = torch.autograd.grad(Uxy, x, is_grads_batched=True, allow_unused=True)[0] # functorch.grad(Uxy, argnums=x, has_aux=False)[0] # , allow_unused=True # x.grad(Uxy)
#         # Eyy = torch.autograd.grad(Vxy, y, is_grads_batched=True, allow_unused=True)[0]
#         # Exy = torch.autograd.grad(Uxy, y, is_grads_batched=True, allow_unused=True)[0] + torch.autograd.grad(Vxy, x, is_grads_batched=True, allow_unused=True)[0] * 0.5 # .detach().numpy()
#
#         Exx = torch.stack(Exx_list, dim=1)
#         Eyy = torch.stack(Eyy_list, dim=1)
#         Exy = torch.stack(Exy_list, dim=1)
#         # Sigma is calculated using the constitutive model equation
#         Sxx = Exx * C11 + Eyy * C12 # σxx = (λ + 2µ)εxx + λ * εyy
#         Syy = Eyy * C11 + Exx * C12 # σyy = (λ + 2µ)εyy + λ * εxx
#         Sxy = Exy * C33 # σxy = 2µ * εxy
#
#         # Loss calculated as the difference between the values of sigma outputted and the given body forces
#         # as described by the momentum balance equation: σij,j = -fi (rearranged from σij,j + fi = 0)
#         Lx = torch.autograd.grad(Sxx, x)[0] + torch.autograd.grad(Sxy, y)[0] - Fx
#         Ly = torch.autograd.grad(Sxy, x)[0] + torch.autograd.grad(Syy, y)[0] - Fy
#
#         return Lx, Ly, Uxy, Vxy

# Data
x_data, y_data = np.meshgrid(np.linspace(0., 1., 20), np.linspace(0., 1., 20)) # creating 400 point grid
x_data = x_data.reshape((-1))
y_data = y_data.reshape((-1))
u_data = np.cos(2*pi*x_data) * np.sin(pi*y_data) # exact solution for Ux
v_data = np.sin(pi*x_data) * Q * y_data**4/4 # exact solution for Uy
# Split the data into training and testing sets - 80% training, 20% testing
x_train, x_test, y_train, y_test, u_train, u_test, v_train, v_test = train_test_split(x_data, y_data, u_data, v_data,
                                                                                      test_size=0.2)


# Training
model = PINN(2, 4*[30], 4*[30])
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
optimizer2 = torch.optim.LBFGS(model.parameters(), lr=0.000001)
mse = torch.nn.MSELoss(reduction='mean') # using mean square error for loss
# mse = mse.cuda()
n_epochs = 75000
changing_point = 0.5
batch_size = 500
mu_list = []
lmbd_list = []
avg_val_loss = []
avg_train_loss = []
avg_val_Lxy_loss = []
avg_train_Lxy_loss = []
avg_val_UxUy_loss = []
avg_train_UxUy_loss = []


for epoch in range(n_epochs):
    if epoch == changing_point * n_epochs:
        optimizer = optimizer2
    n_batches = np.ceil(x_train.shape[0]/batch_size).astype(int) # number of batches
    model.train()
    train_loss_list = []
    val_loss_list = []
    train_Lxy_loss_list = []
    train_UxUy_loss_list = []
    val_Lxy_loss_list = []
    val_UxUy_loss_list = []

    for batch_i in range(n_batches):
        batch_start = batch_i * batch_size
        batch_stop = np.minimum(x_train.shape[0], (batch_i + 1) * batch_size)

        x_batch = torch.from_numpy(x_train[batch_start:batch_stop]).to(dtype=torch.float32)
        y_batch = torch.from_numpy(y_train[batch_start:batch_stop]).to(dtype=torch.float32)
        u_batch = torch.from_numpy(u_train[batch_start:batch_stop]).to(dtype=torch.float32)
        v_batch = torch.from_numpy(v_train[batch_start:batch_stop]).to(dtype=torch.float32)

        Lx_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        Ly_batch = torch.from_numpy(np.zeros_like(y_batch)).to(dtype=torch.float32)

        x_batch.requires_grad = True
        y_batch.requires_grad = True


        def closure():
            optimizer.zero_grad()
            Lx_hat, Ly_hat, Uxy_hat, Vxy_hat = model(x_batch, y_batch)

            Lx_loss = mse(Lx_hat, Lx_batch)
            Ly_loss = mse(Ly_hat, Ly_batch)
            Uxy_loss = mse(Uxy_hat, u_batch)
            Vxy_loss = mse(Vxy_hat, v_batch)

            train_Lxy_loss_list.append(Lx_loss.item() + Ly_loss.item())
            train_UxUy_loss_list.append(Uxy_loss.item() + Vxy_loss.item())

            total_loss = (Lx_loss + Ly_loss) * 0.5 + Uxy_loss + Vxy_loss
            train_loss_list.append(total_loss.item())
            total_loss.backward()

            return total_loss


        if epoch >= changing_point * n_epochs:
            optimizer.step(closure)

        else:
            closure()
            optimizer.step()
        # closure()
        # optimizer.step()

    avg_train_loss.append(np.mean(train_loss_list))
    avg_train_Lxy_loss.append(np.mean(train_Lxy_loss_list))
    avg_train_UxUy_loss.append(np.mean(train_UxUy_loss_list))

    # Validate
    model.eval()

    n_batches_val = np.ceil(x_test.shape[0] / batch_size).astype(int)

    for batch_i in range(n_batches_val):
        batch_start = batch_i * batch_size
        batch_stop = np.minimum(x_test.shape[0], (batch_i + 1) * batch_size)

        x_batch = torch.from_numpy(x_test[batch_start:batch_stop]).to(dtype=torch.float32)
        y_batch = torch.from_numpy(y_test[batch_start:batch_stop]).to(dtype=torch.float32)
        u_batch = torch.from_numpy(u_test[batch_start:batch_stop]).to(dtype=torch.float32)
        v_batch = torch.from_numpy(v_test[batch_start:batch_stop]).to(dtype=torch.float32)
        Lx_batch = torch.from_numpy(np.zeros_like(x_batch)).to(dtype=torch.float32)
        Ly_batch = torch.from_numpy(np.zeros_like(y_batch)).to(dtype=torch.float32)

        x_batch.requires_grad = True
        y_batch.requires_grad = True

        Lx_hat, Ly_hat, Uxy_hat, Vxy_hat = model(x_batch, y_batch)

        Lx2_loss = mse(Lx_hat, Lx_batch)
        Ly2_loss = mse(Ly_hat, Ly_batch)
        Uxy2_loss = mse(Uxy_hat, u_batch)
        Vxy2_loss = mse(Vxy_hat, v_batch)

        val_Lxy_loss_list.append(Lx2_loss.item() + Ly2_loss.item())
        val_UxUy_loss_list.append(Uxy2_loss.item() + Vxy2_loss.item())

        total_loss2 = Lx2_loss + Ly2_loss + Uxy2_loss + Vxy2_loss
        val_loss_list.append(total_loss2.item())

    avg_val_loss.append(np.mean(val_loss_list))
    avg_val_Lxy_loss.append(np.mean(val_Lxy_loss_list))
    avg_val_UxUy_loss.append(np.mean(val_UxUy_loss_list))

    if epoch % 50 == 0:
        mu_list.append(model.mu)
        lmbd_list.append(model.lmbd)
        print("Epoch " + str(epoch) + ": Training loss: " + str(avg_train_loss[epoch]))
        print("Epoch " + str(epoch) + ": Validation loss: " + str(avg_val_loss[epoch]))
        print("Mu: " + str(model.mu.item()) + " Lambda: " + str(model.lmbd.item()))
        # print(f'Training: Lx loss {Lx_loss}, Ly loss {Ly_loss}, Uxy loss {Uxy_loss}, Vxy loss {Vxy_loss} ')
        # print(f'Validation: Lx loss {Lx2_loss}, Ly loss {Ly2_loss}, Uxy loss {Uxy2_loss}, Vxy loss {Vxy2_loss} ')
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

plt.plot(avg_train_Lxy_loss, label='LxLy Train Loss')
plt.plot(avg_train_UxUy_loss, label='UxUy Train Loss')
plt.legend()
plt.show()

plt.plot(avg_val_Lxy_loss, label='LxLy Val Loss')
plt.plot(avg_val_UxUy_loss, label='UxUy Val Loss')
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

