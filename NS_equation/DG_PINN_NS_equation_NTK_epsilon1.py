# =============================================================================
# IMPORT MODULES
# =============================================================================
import torch
import scipy.io
import numpy as np
import torch.nn as nn
from scipy.io import savemat
from torch.autograd import grad
from timeit import default_timer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

# =============================================================================
# CREATE PHYSICS-INFORMED NEURAL NETWORK
# =============================================================================

class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=50, num_hidden=3, activation='sin'):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.epoch = 0
        self.alpha1 = nn.Parameter(torch.tensor([alpha], requires_grad=True).float())
        self.alpha2 = nn.Parameter(torch.tensor([alpha], requires_grad=True).float())

        if activation == 'sin':
            self.activation = torch.sin
        elif activation == 'tanh':
            self.activation = torch.tanh

    def forward(self, x, y, t):
        out = torch.cat([x, y, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    # Define Navier Stokes Equations (Time-dependent PDEs)
    def Navier_Stokes_Eq(self, x, y, t):
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True
        pred = self.forward(x, y, t)
        u = pred[:,0:1]
        v = pred[:,1:2]
        p = pred[:,2:3]
        du_x = grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_y = grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_t = grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

        dv_x = grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        dv_y = grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        dv_t = grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

        dp_x = grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        dp_y = grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        du_xx = grad(du_x, x, grad_outputs=torch.ones_like(du_x), create_graph=True)[0]
        du_yy = grad(du_y, y, grad_outputs=torch.ones_like(du_x), create_graph=True)[0]
        dv_xx = grad(dv_x, x, grad_outputs=torch.ones_like(du_y), create_graph=True)[0]
        dv_yy = grad(dv_y, y, grad_outputs=torch.ones_like(du_y), create_graph=True)[0]
        continuity = du_x + dv_y
        x_momentum = du_t + self.alpha1 * (u * du_x + v * du_y) + dp_x - self.alpha2 * (du_xx + du_yy)
        y_momentum = dv_t + self.alpha1 * (u * dv_x + v * dv_y) + dp_y - self.alpha2 * (dv_xx + dv_yy)
        return [continuity, x_momentum, y_momentum]

    # Define the loss function
    def loss_PDE(self, x, y, t):
        continuity, x_momentum, y_momentum = self.Navier_Stokes_Eq(x, y, t)
        #loss_r = torch.mean(continuity**2 + x_momentum**2 + y_momentum**2)
        loss_r1 = torch.mean(x_momentum**2)
        loss_r2 = torch.mean(y_momentum**2)
        loss_r3 = torch.mean(continuity**2)
        return loss_r1, loss_r2, loss_r3
        
    def loss_data(self, x, y, t, u, v):
        pred = self.forward(x, y, t)
        u_pred = pred[:,0:1]
        v_pred = pred[:,1:2]
        p_pred = pred[:,2:3]
        loss_du = torch.mean((u - u_pred)**2)
        loss_dv = torch.mean((v - v_pred)**2)
        # loss_d = loss_du + loss_dv + loss_dp
        return loss_du, loss_dv

    # Prediction
    def model_test(self, x, y, t, u, v):               
        pred = self.forward(x, y, t)
        u_pred = pred[:,0:1]
        v_pred = pred[:,1:2]
        # Relative L2 Norm of the error (Vector)  
        error_vec_u = torch.linalg.norm((u - u_pred),2) / torch.linalg.norm(u, 2)    
        error_vec_v = torch.linalg.norm((v - v_pred),2) / torch.linalg.norm(v, 2)      
        return error_vec_u, error_vec_v
    
def closure():
    global lambda1, lambda2, lambda3, lambda4, lambda5

    # Zero gradients
    optimizer.zero_grad()

    # Calculate losses
    loss_r1, loss_r2, loss_r3 = model.loss_PDE(X_train['PDE']['x'], X_train['PDE']['y'], X_train['PDE']['t'])
    loss_du, loss_dv = model.loss_data(X_train['PDE']['x'], X_train['PDE']['y'], X_train['PDE']['t'], 
                            X_train['PDE']['u'], X_train['PDE']['v'])

    # Calculate total loss with updated alpha1
    loss = lambda1*loss_r1 + lambda2*loss_r2 + lambda3*loss_r3 + lambda4*loss_du + lambda5*loss_dv 

    # Backpropagation
    loss.backward()

    # Append losses for monitoring
    epoch_loss_r1.append(loss_r1.item())
    epoch_loss_r2.append(loss_r2.item())
    epoch_loss_r3.append(loss_r3.item())
    epoch_loss_du.append(loss_du.item())
    epoch_loss_dv.append(loss_dv.item())
    epoch_alpha1.append(model.alpha1.item())
    epoch_alpha2.append(model.alpha2.item())
    epoch_lambda1.append(lambda1)
    epoch_lambda2.append(lambda2)
    epoch_lambda3.append(lambda3)
    epoch_lambda4.append(lambda4)
    epoch_lambda5.append(lambda5)
    # # Print losses infrequently to avoid slowing down training
    # if model.epoch % 10000 == 0:
    #     print('Epoch %d, loss = %e, loss_r = %e, loss_d = %e, alpha1 = %e, alpha2 = %e, lambda1= %e , lambda2= %e' %
    #           (iter_1 + model.epoch, float(loss), float(loss_r), float(loss_d), model.alpha1.item(), 
    #            model.alpha2.item(), float(lambda1), float(lambda2)))
    
    model.epoch += 1
    return loss

def train_dg_pinn(model, optimizer, X_train, iters=50001, stopping_loss=1e-2):
    for epoch in range(iters):
        t1 = default_timer()
        optimizer.zero_grad()
        loss_du, loss_dv = model.loss_data(X_train['PDE']['x'], X_train['PDE']['y'], X_train['PDE']['t'], 
                            X_train['PDE']['u'], X_train['PDE']['v'])
        loss = loss_du + loss_dv
        loss.backward()
        optimizer.step()
        epoch_loss_du.append(loss_du.item())
        epoch_loss_dv.append(loss_dv.item())
        t2 = default_timer()
        # if epoch % 1000 == 0:
        #     print('Epoch %d, time = %e, loss = %e, loss_du = %e, loss_dv = %e, loss_dp = %e,  alpha1 = %e,  alpha2 = %e' %
        #           (epoch, float(t2-t1), float(loss),  float(loss_du),  float(loss_dv),   model.alpha1.item(),  model.alpha2.item()))
            
        error_vec_u, error_vec_v = model.model_test(X_validation['x'], X_validation['y'], X_validation['t'],
                                                                    X_validation['u'], X_validation['v'])
        rl2_err = (error_vec_u + error_vec_v)/2
        # Stopping condition
        if rl2_err < stopping_loss:
            print(f'Stopping early at epoch {epoch} as relative l2 error fell below {stopping_loss}')
            break

def train_pinn(model, optimizer, X_train, NTK, iters=50001, stopping_loss=1e-2):
    for epoch in range(iters):
        global lambda1, lambda2, lambda3, lambda4, lambda5
        # Zero gradients
        optimizer.zero_grad()

        # Calculate losses
        loss_r1, loss_r2, loss_r3 = model.loss_PDE(X_train['PDE']['x'], X_train['PDE']['y'], X_train['PDE']['t'])
        loss_du, loss_dv = model.loss_data(X_train['PDE']['x'], X_train['PDE']['y'], X_train['PDE']['t'], 
                                X_train['PDE']['u'], X_train['PDE']['v'])

        # Calculate total loss with updated alpha1
        loss = lambda1*loss_r1 + lambda2*loss_r2 + lambda3*loss_r3 + lambda4*loss_du + lambda5*loss_dv 

        # Backpropagation
        loss.backward()

        # Append losses for monitoring
        epoch_loss_r1.append(loss_r1.item())
        epoch_loss_r2.append(loss_r2.item())
        epoch_loss_r3.append(loss_r3.item())
        epoch_loss_du.append(loss_du.item())
        epoch_loss_dv.append(loss_dv.item())
        epoch_alpha1.append(model.alpha1.item())
        epoch_alpha2.append(model.alpha2.item())
        epoch_lambda1.append(lambda1)
        epoch_lambda2.append(lambda2)
        epoch_lambda3.append(lambda3)
        epoch_lambda4.append(lambda4)
        epoch_lambda5.append(lambda5)

        error_vec_u, error_vec_v = model.model_test(X_validation['x'], X_validation['y'], X_validation['t'],
                                                                    X_validation['u'], X_validation['v'])
        rl2_err = (error_vec_u + error_vec_v)/2
        # Stopping condition
        if rl2_err < stopping_loss:
            print(f'Stopping early at epoch {epoch} as relative l2 error fell below {stopping_loss}')
            break

        if NTK == 'True' and epoch % 1000 == 0:
            lambda1, lambda2, lambda3, lambda4, lambda5 = Adap_weights(model, X_train)

# Load training data
def get_data(batch_sizes):
    data = scipy.io.loadmat("../Data/cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    # choose number of training points
    # Total points
    total_points = data_domain.shape[0]
    # Define indices for iPDE, and data points
    ids_total = np.arange(total_points)
    # Random selection function
    def random_selection(id_set, size):
        np.random.seed(seeds_num)
        return np.random.choice(id_set, size, replace=False)

    # Select random points for each category
    id_pde = random_selection(ids_total, batch_sizes['PDE'])
    # id_data = random_selection(np.arange(total_points), batch_sizes['data'])
    id_data = random_selection(ids_total, batch_sizes['data'])

    # Function to convert indices to tensor
    def to_tensor(ids):
        return {
            'x': torch.from_numpy(data_domain[ids, 0:1]).float().to(device),
            'y': torch.from_numpy(data_domain[ids, 1:2]).float().to(device),
            't': torch.from_numpy(data_domain[ids, 2:3]).float().to(device),
            'u': torch.from_numpy(data_domain[ids, 3:4]).float().to(device),
            'v': torch.from_numpy(data_domain[ids, 4:5]).float().to(device),
            'p': torch.from_numpy(data_domain[ids, 5:6]).float().to(device),
        }
    
    X_train = {'PDE': to_tensor(id_pde), 'data': to_tensor(id_data)}

    all_train_ids = np.union1d(id_pde, id_data)
    # Create the validation set
    id_remaining = np.setdiff1d(np.arange(total_points), all_train_ids)
    id_validation = random_selection(id_remaining, batch_sizes['validation'])

    # Update id_data to exclude validation points for the test set
    all_used_ids = np.union1d(id_validation, np.union1d(id_pde, id_data))

    # Create a boolean mask for all points, then exclude the training indices
    mask = np.ones(total_points, dtype=bool)
    mask[all_used_ids] = False

    X_validation = to_tensor(id_validation)
    X_test = to_tensor(mask)
    X_true = to_tensor(ids_total)

    return X_train, X_validation, X_test, X_true

def Adap_weights(model, X_train):
    # Zero out gradients
    model.zero_grad()
    # Get all parameters excluding those containing "lambda1" in their names
    param_tensors = [param for param in model.parameters()]
    #Move model parameters to CPU
    param_tensors_cpu = [param.cpu() for param in model.parameters()]
    
    x = X_train['PDE']['x'].requires_grad_()
    y = X_train['PDE']['y'].requires_grad_()
    t = X_train['PDE']['t'].requires_grad_()

    # Divide data into batches
    batch_size = 200
    total_data = x.shape[0]
    batches = (total_data + batch_size - 1) // batch_size

    # Initialize Jacobian matrix
    jacobian_r1 = torch.zeros(total_data, sum(p.numel() for p in param_tensors_cpu))
    jacobian_r2 = torch.zeros(total_data, sum(p.numel() for p in param_tensors_cpu))
    jacobian_r3 = torch.zeros(total_data, sum(p.numel() for p in param_tensors_cpu))

    for batch in range(batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, total_data)

        # Select batch data
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        t_batch = t[start_idx:end_idx]
        
        # Compute model predictions and derivatives for the batch
        pred = model(x_batch, y_batch, t_batch)
        u = pred[:,0:1]
        v = pred[:,1:2]
        p = pred[:,2:3]
        du_x = grad(u, x_batch, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_y = grad(u, y_batch, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_t = grad(u, t_batch, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]

        dv_x = grad(v, x_batch, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        dv_y = grad(v, y_batch, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        dv_t = grad(v, t_batch, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

        dp_x = grad(p, x_batch, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        dp_y = grad(p, y_batch, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

        du_xx = grad(du_x, x_batch, grad_outputs=torch.ones_like(du_x), create_graph=True)[0]
        du_yy = grad(du_y, y_batch, grad_outputs=torch.ones_like(du_x), create_graph=True)[0]
        dv_xx = grad(dv_x, x_batch, grad_outputs=torch.ones_like(du_y), create_graph=True)[0]
        dv_yy = grad(dv_y, y_batch, grad_outputs=torch.ones_like(du_y), create_graph=True)[0]

        continuity = du_x + dv_y
        x_momentum = du_t + model.alpha1 * (u * du_x + v * du_y) + dp_x - model.alpha2 * (du_xx + du_yy)
        y_momentum = dv_t + model.alpha1 * (u * dv_x + v * dv_y) + dp_y - model.alpha2 * (dv_xx + dv_yy)

        # Now process residuals_batch
        for i in range(len(continuity)):
            grad_outputs = torch.zeros_like(continuity)
            grad_outputs[i] = 1
            # Compute gradients and immediately move them to CPU
            grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(x_momentum, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
            grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
            jacobian_r1[start_idx + i] = grad_row

            grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(y_momentum, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
            grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
            jacobian_r2[start_idx + i] = grad_row

            grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(continuity, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
            grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
            jacobian_r3[start_idx + i] = grad_row
   

    x = X_train['data']['x'].requires_grad_()
    y = X_train['data']['y'].requires_grad_()
    t = X_train['data']['t'].requires_grad_()

    # Divide data into batches
    total_data = x.shape[0]
    batch_size = 200
    batches = (total_data + batch_size - 1) // batch_size

    # Initialize Jacobian matrix
    jacobian_du = torch.zeros(total_data, sum(p.numel() for p in param_tensors_cpu))
    jacobian_dv = torch.zeros(total_data, sum(p.numel() for p in param_tensors_cpu))

    for batch in range(batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, total_data)
        
        # Select batch data
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        t_batch = t[start_idx:end_idx]
        
        # Compute model predictions and derivatives for the batch
        pred = model(x_batch, y_batch, t_batch)
        u_pred = pred[:,0:1]
        v_pred = pred[:,1:2]
        # Now process residuals_batch
        for i in range(len(u_pred)):
            grad_outputs = torch.zeros_like(u_pred)
            grad_outputs[i] = 1
            # Compute gradients and immediately move them to CPU
            grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(u_pred, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
            grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
            jacobian_du[start_idx + i] = grad_row

            grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(v_pred, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
            grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
            jacobian_dv[start_idx + i] = grad_row

    # Adapative weightings
    Krr11 = jacobian_r1 @ jacobian_r1.T
    Krr22 = jacobian_r2 @ jacobian_r2.T
    Krr33 = jacobian_r3 @ jacobian_r3.T
    Kdduu = jacobian_du @ jacobian_du.T
    Kddvv = jacobian_dv @ jacobian_dv.T

    K_trace = (torch.trace(Krr11) + torch.trace(Krr22) + torch.trace(Krr33))/batch_sizes['PDE'] + \
            (torch.trace(Kdduu) + torch.trace(Kddvv))/batch_sizes['data']
    epsilon = 1e-8  # Small regularization term
    lambda_1 = K_trace / (torch.trace(Krr11)/batch_sizes['PDE'] + epsilon)
    lambda_2 = K_trace / (torch.trace(Krr22)/batch_sizes['PDE'] + epsilon)
    lambda_3 = K_trace / (torch.trace(Krr33)/batch_sizes['PDE'] + epsilon)
    lambda_4 = K_trace / (torch.trace(Kdduu)/batch_sizes['data'] + epsilon) 
    lambda_5 = K_trace / (torch.trace(Kddvv)/batch_sizes['data'] + epsilon)

    lambda1 = (lambda_1.item())
    lambda2 = (lambda_2.item())
    lambda3 = (lambda_3.item())
    lambda4 = (lambda_4.item())
    lambda5 = (lambda_5.item())

    return lambda1, lambda2, lambda3, lambda4, lambda5

# Compute the relative L2 error
def relative_l2_error(pred, true):
    return torch.norm(pred - true) / torch.norm(true)

# =============================================================================
# DATA & PARAMETERS
# =============================================================================

seeds_num = 666
torch.manual_seed(seeds_num)
np.random.seed(seeds_num)
alpha = np.random.rand()
batch_sizes = {'initial': 100, 'bounds': 200, 'PDE': 1000, 'data': 10000, 'validation': 10000}
stopping_conditions = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
             0.009, 0.008, 0.007, 0.006, 0.005]
iter_1 = 200000 # Maximun number of iterations for Adam optimizer
iter_2 = 20000  # Maximun number  of iterations for L-BFGS optimizer
X_train, X_validation, X_test, X_true = get_data(batch_sizes)

# =============================================================================
# TRAIN MODEL
# =============================================================================
for epsilon_1 in stopping_conditions:
    torch.manual_seed(seeds_num)
    epoch_loss_r1 = []
    epoch_loss_r2 = []
    epoch_loss_r3 = []
    epoch_loss_du = []
    epoch_loss_dv = []
    epoch_alpha1= []
    epoch_alpha2= []
    epoch_lambda1 = []
    epoch_lambda2 = []
    epoch_lambda3 = []
    epoch_lambda4 = []
    epoch_lambda5 = []

    model = PINN(
        input_dim=3,
        output_dim=3,
        hidden_dim=100,
        num_hidden=3, 
        activation='sin'
    ).to(device)
    print(model)

    t11 = default_timer()
    # Adam optimizer to decrease loss in Phase 1
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    train_dg_pinn(model, optimizer, X_train, iters=iter_1, stopping_loss=epsilon_1)

    lambda1, lambda2, lambda3, lambda4, lambda5 = Adap_weights(model, X_train)

    # L-BFGS optimizer for fine-tuning in Phase 2
    optimizer = torch.optim.LBFGS(list(model.parameters()), lr=1e-1, max_iter=1,
                                history_size=100)
    for epoch in range(iter_2):
        optimizer.zero_grad()
        optimizer.step(closure)

    t22 = default_timer()
    print('Time elapsed: %.2f min' % ((t22 - t11) / 60))

    # =============================================================================
    # SAVE DATA & MODEL
    # =============================================================================

    error_vec_u, error_vec_v = model.model_test(X_test['x'], X_test['y'], X_test['t'], X_test['u'], X_test['v'])
    print('Test Error: %.5f'  % (error_vec_u))
    print('Test Error: %.5f'  % (error_vec_v))

    pred = model(X_test['x'], X_test['y'], X_test['t'])

    u_test = X_test['u'].cpu().detach().numpy()
    v_test = X_test['v'].cpu().detach().numpy()
    p_test = X_test['p'].cpu().detach().numpy()

    u_pred0 = pred[:,0:1].cpu().detach().numpy()
    v_pred0 = pred[:,1:2].cpu().detach().numpy()
    p_pred0 = pred[:,2:3].cpu().detach().numpy()

    pred = model(X_true['x'], X_true['y'], X_true['t'])

    u_true = X_true['u'].cpu().detach().numpy()
    v_true = X_true['v'].cpu().detach().numpy()
    p_true = X_true['p'].cpu().detach().numpy()

    u_pred = pred[:,0:1].cpu().detach().numpy()
    v_pred = pred[:,1:2].cpu().detach().numpy()
    p_pred = pred[:,2:3].cpu().detach().numpy()

    # Save data and model
    savemat(f'dgpinn_NS_NTK_epsilon_{epsilon_1}.mat',
            {'u_pred0': u_pred0, 'v_pred0': v_pred0, 'p_pred0': p_pred0, 'u_test': u_test, 'v_test': v_test, 'p_test': p_test, 
             'u_pred': u_pred, 'v_pred': v_pred, 'p_pred': p_pred, 'u_true': u_true, 'v_true': v_true, 'p_true': p_true, 
            'loss_r1': epoch_loss_r1, 'loss_r2': epoch_loss_r2, 'loss_r3': epoch_loss_r3, 
            'loss_du': epoch_loss_du, 'loss_dv': epoch_loss_dv, 
            'lambd1': epoch_lambda1, 'lambd2': epoch_lambda2, 'lambd3': epoch_lambda3, 
            'lambd4': epoch_lambda4, 'lambd5': epoch_lambda5, 
            'alpha1': epoch_alpha1, 'alpha2': epoch_alpha2, 'time': t22 - t11})
    