# =============================================================================
# IMPORT MODULES
# =============================================================================
import torch
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
    def __init__(self, input_dim=3, output_dim=2, hidden_dim=50, num_hidden=3, activation='sin'):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.epoch = 0
        self.alpha = nn.Parameter(torch.tensor([alpha], requires_grad=True).float())

        if activation == 'sin':
            self.activation = torch.sin
        elif activation == 'tanh':
            self.activation = torch.tanh

    def forward(self, x, t):
        out = torch.cat([x, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def loss_PDE(self, x, t):
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred = self.forward(x, t)
        
        u_x = grad(u_pred, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        u_xxx = grad(u_xx, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        u_xxxx = grad(u_xxx, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        
        u_t = grad(u_pred, t, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        u_tt = grad(u_t, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]

        residual = u_tt + self.alpha**2*u_xxxx
        loss_r = torch.mean(residual ** 2)
        return loss_r

    def loss_initial(self, x, t):    
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred = self.forward(x, t)
    
        u_t = grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        
        loss_i = torch.mean(u_t ** 2)
        return loss_i
    
    def loss_bounds(self, x, t):    
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred = self.forward(x, t)
        u_x = grad(u_pred, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

        loss_b = torch.mean(u_xx ** 2)
        return loss_b
        
    def loss_data(self, x, t, u):    
        u_pred = self.forward(x, t) 
        loss_d = torch.mean((u_pred - u) ** 2)
        return loss_d

    def losses(self, X_train):
        loss_r = self.loss_PDE(X_train['PDE']['x'], X_train['PDE']['t'])
        loss_i = self.loss_initial(X_train['initial']['x'], X_train['initial']['t'])
        loss_b = self.loss_bounds(X_train['bounds']['x'], X_train['bounds']['t'])
        loss_d = self.loss_data(X_train['data']['x'], X_train['data']['t'], X_train['data']['u']) + \
                    self.loss_data(X_train['initial']['x'], X_train['initial']['t'], X_train['initial']['u']) + \
                    self.loss_data(X_train['bounds']['x'], X_train['bounds']['t'], X_train['bounds']['u'])
        return loss_r, loss_i, loss_b, loss_d/3
    
def closure():
    global lambda1, lambda2, lambda3, lambda4

    # Zero gradients
    optimizer.zero_grad()

    # Calculate losses
    loss_r, loss_i, loss_b, loss_d = model.losses(X_train)

    # Calculate total loss with updated alpha1
    loss = lambda1*loss_r + lambda2*loss_i + lambda3*loss_b + lambda4*loss_d

    # Backpropagation
    loss.backward()

    # Append losses for monitoring
    epoch_loss_r.append(loss_r.item())
    epoch_loss_i.append(loss_i.item())
    epoch_loss_b.append(loss_b.item())
    epoch_loss_d.append(loss_d.item())
    epoch_alpha.append(model.alpha.item())
    epoch_lambda1.append(lambda1)
    epoch_lambda2.append(lambda2)
    epoch_lambda3.append(lambda3)
    epoch_lambda4.append(lambda4)
    # Print losses infrequently to avoid slowing down training
    if model.epoch % 10000 == 0:
        print('Epoch %d, loss = %e, loss_r = %e, loss_i = %e, loss_b = %e, loss_d = %e, alpha = %e, lambda1= %e , lambda2= %e, lambda3= %e, lambda4=%e' %
              (iter_1 + model.epoch, float(loss), float(loss_r), float(loss_i), float(loss_b), float(loss_d),
                model.alpha.item(), float(lambda1), float(lambda2), float(lambda3), float(lambda4)))
    
    model.epoch += 1
    return loss

def train_dg_pinn(model, optimizer, X_train, iters=50001, stopping_loss=1e-2):
    for epoch in range(iters):
        t1 = default_timer()
        optimizer.zero_grad()
        loss_d = model.loss_data(X_train['data']['x'], X_train['data']['t'], X_train['data']['u'])
        loss = loss_d
        loss.backward()
        optimizer.step()
        epoch_loss_d.append(loss_d.item())
        t2 = default_timer()
        if epoch % 10000 == 0:
            print('Epoch %d, time = %e, loss = %e,  alpha = %e' %
                  (epoch, float(t2-t1), float(loss),  model.alpha.item()))
            
        u_pred = model(X_validation['x'], X_validation['t'])
        # Calculate relative L2 errors
        u_error = relative_l2_error(u_pred, X_validation['u'])
        # Stopping condition
        if u_error < stopping_loss:
            print(f'Stopping early at epoch {epoch} as relative l2 error fell below {stopping_loss}')
            break

def train_pinn(model, optimizer, X_train, NTK, iters=50001, stopping_loss=1e-2):
    for epoch in range(iters):
        global lambda1, lambda2, lambda3, lambda4
        t1 = default_timer()
        optimizer.zero_grad()
        # Calculate losses
        loss_r, loss_i, loss_b, loss_d = model.losses(X_train)
    
        # Calculate total loss with updated alpha1
        loss = lambda1*loss_r + lambda2*loss_i + lambda3*loss_b + lambda4*loss_d
        loss.backward()
        optimizer.step()

        # Append losses for monitoring
        epoch_loss_r.append(loss_r.item())
        epoch_loss_i.append(loss_i.item())
        epoch_loss_b.append(loss_b.item())
        epoch_loss_d.append(loss_d.item())
        epoch_alpha.append(model.alpha.item())
        epoch_lambda1.append(lambda1)
        epoch_lambda2.append(lambda2)
        epoch_lambda3.append(lambda3)
        epoch_lambda4.append(lambda4)
        t2 = default_timer()
        if epoch % 10000 == 0:
            print('Epoch %d, time = %e, loss = %e, loss_r = %e, loss_i = %e, loss_b = %e, loss_d = %e, alpha = %e, lambda1= %e , lambda2= %e, lambda3= %e, lambda4=%e' %
                  (epoch, float(t2-t1), float(loss), float(loss_r), float(loss_i), float(loss_b), float(loss_d),
                    model.alpha.item(), float(lambda1), float(lambda2), float(lambda3), float(lambda4)))

        if NTK == 'True' and epoch % 1000 == 0:
            lambda1, lambda2, lambda3, lambda4 = Adap_weights(model, X_train)

        u_pred = model(X_validation['x'], X_validation['t'])
        # Calculate relative L2 errors
        u_error = relative_l2_error(u_pred, X_validation['u'])
        # if current_loss < stopping_loss:
        if u_error < stopping_loss:
            print(f'Stopping early at epoch {epoch} as relative l2 error fell below {stopping_loss}')
            break

def add_noise(signal, snr_db):
    # Calculate signal power and convert SNR from dB
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)

    # Calculate noise power and generate noise
    noise_power = signal_power / snr_linear
    np.random.seed(seeds_num)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to the signal
    return signal + noise

def get_data(alpha, batch_sizes, snr_db):
    # Load and scale data
    x = np.linspace(0, 1, 201)
    t = np.linspace(0, 1, 201)
    X, T = np.meshgrid(x, t)
    U = np.sin(np.pi*X)*np.cos(np.pi**2*alpha*T)

    # Flatten arrays
    x_true = X.flatten('C')[:, None]
    t_true = T.flatten('C')[:, None]
    u_true = U.flatten('C')[:, None]
    u_noisy = add_noise(u_true, snr_db)  # For initial points in training

    # Total points
    total_points = len(x) * len(t)

    # Define indices for initial, boundary, PDE, and data points
    id_initial = np.where(t_true == 0)[0]
    id_bounds = np.where((x_true == 0) | (x_true == 1))[0]
    id_pde = np.setdiff1d(np.arange(total_points), np.union1d(id_initial, id_bounds))
    id_data = np.setdiff1d(np.arange(total_points), np.union1d(id_initial, id_bounds))

    # Random selection function
    def random_selection(id_set, size):
        np.random.seed(seeds_num)
        return np.random.choice(id_set, size, replace=False)

    # Select random points for each category
    id_initial = random_selection(id_initial, batch_sizes['initial'])
    id_bounds = random_selection(id_bounds, batch_sizes['bounds'])
    id_pde = random_selection(id_pde, batch_sizes['PDE'])
    # id_data = random_selection(np.arange(total_points), batch_sizes['data'])
    id_data = random_selection(id_data, batch_sizes['data'])

    # Function to convert indices to tensor
    def to_tensor(ids):
        return {
            'x': torch.from_numpy(x_true[ids, :]).float().to(device),
            't': torch.from_numpy(t_true[ids, :]).float().to(device),
            'u': torch.from_numpy(u_noisy[ids, :]).float().to(device)
        }

    # Create dictionaries
    X_train = {'initial': to_tensor(id_initial), 'bounds': to_tensor(id_bounds), 
               'PDE': to_tensor(id_pde), 'data': to_tensor(id_data)}

    all_train_ids = np.union1d(id_initial, np.union1d(id_bounds, np.union1d(id_pde, id_data)))
    # Create the validation set
    id_remaining = np.setdiff1d(np.arange(total_points), all_train_ids)
    id_validation = random_selection(id_remaining, batch_sizes['validation'])

    # Update id_data to exclude validation points for the test set
    all_used_ids = np.union1d(id_validation, np.union1d(id_initial, np.union1d(id_bounds, np.union1d(id_pde, id_data))))

    # Create a boolean mask for all points, then exclude the training indices
    mask = np.ones(total_points, dtype=bool)
    mask[all_used_ids] = False

    X_validation = {'x': torch.from_numpy(x_true[id_validation, :]).float().to(device),
                    't': torch.from_numpy(t_true[id_validation, :]).float().to(device),
                    'u': torch.from_numpy(u_noisy[id_validation, :]).float().to(device)}

    # Use the mask to select the remaining points for X_test
    X_test = {'x': torch.from_numpy(x_true[mask, :]).float().to(device),
              't': torch.from_numpy(t_true[mask, :]).float().to(device),
              'u': torch.from_numpy(u_true[mask, :]).float().to(device)}
    
    X_true = {'x': torch.from_numpy(x_true).float().to(device),
              't': torch.from_numpy(t_true).float().to(device),
              'u': torch.from_numpy(u_true).float().to(device)}

    return u_noisy, X, T, U, X_train, X_validation, X_test, X_true

def Adap_weights(model, X_train):
    # Zero out gradients
    model.zero_grad()
    # Get all parameters excluding those containing "lambda1" in their names
    param_tensors = [param for param in model.parameters()]
    #Move model parameters to CPU
    param_tensors_cpu = [param.cpu() for param in model.parameters()]
    
    x = X_train['PDE']['x'].requires_grad_()
    t = X_train['PDE']['t'].requires_grad_()

    # Divide data into batches
    batch_size = 200
    total_data = x.shape[0]
    batches = (total_data + batch_size - 1) // batch_size

    # Initialize Jacobian matrix
    jacobian_r = torch.zeros(total_data, sum(p.numel() for p in param_tensors_cpu))

    for batch in range(batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, total_data)

        # Select batch data
        x_batch = x[start_idx:end_idx]
        t_batch = t[start_idx:end_idx]
        
        # Compute model predictions and derivatives for the batch
        u_pred = model(x_batch, t_batch)
        
        # Compute model predictions and derivatives for the batch
        u_x = grad(u_pred, x_batch, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        u_xx = grad(u_x, x_batch, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        u_xxx = grad(u_xx, x_batch, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        u_xxxx = grad(u_xxx, x_batch, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        
        u_t = grad(u_pred, t_batch, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        u_tt = grad(u_t, t_batch, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]

        residuals = u_tt + model.alpha**2*u_xxxx

        # Now process residuals_batch
        for i in range(len(residuals)):
            grad_outputs = torch.zeros_like(residuals)
            grad_outputs[i] = 1
            # Compute gradients and immediately move them to CPU
            grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(residuals, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
            grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
            jacobian_r[start_idx + i] = grad_row

    x = X_train['initial']['x'].requires_grad_()
    t = X_train['initial']['t'].requires_grad_()
    u_pred = model(x, t)
    u_t = grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    # Create jacobian matrix of data loss on CPU
    jacobian_i = torch.zeros(len(u_t), sum(p.numel() for p in param_tensors)).cpu()

    for i in range(len(u_t)):
        grad_outputs = torch.zeros_like(u_t)
        grad_outputs[i] = 1
        # Compute gradients and immediately move them to CPU
        grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(u_t, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
        grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
        jacobian_i[i] = grad_row

    x = X_train['bounds']['x'].requires_grad_()
    t = X_train['bounds']['t'].requires_grad_()
    u_pred = model(x, t)
    u_x = grad(u_pred, x, grad_outputs=torch.ones_like(x), retain_graph=True, create_graph=True)[0]
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(x),  create_graph=True)[0]

    # Create jacobian matrix of data loss on CPU
    jacobian_b = torch.zeros(len(u_xx), sum(p.numel() for p in param_tensors)).cpu()

    for i in range(len(u_xx)):
        grad_outputs = torch.zeros_like(u_xx)
        grad_outputs[i] = 1
        # Compute gradients and immediately move them to CPU
        grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(u_xx, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
        grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
        jacobian_b[i] = grad_row

    x = torch.cat([X_train['data']['x'].requires_grad_(), X_train['initial']['x'].requires_grad_(), X_train['bounds']['x'].requires_grad_()])
    t = torch.cat([X_train['data']['t'].requires_grad_(), X_train['initial']['t'].requires_grad_(), X_train['bounds']['t'].requires_grad_()])
    
    # Divide data into batches
    total_data = x.shape[0]
    batch_size = 100
    batches = (total_data + batch_size - 1) // batch_size

    # Initialize Jacobian matrix
    jacobian_d = torch.zeros(total_data, sum(p.numel() for p in param_tensors_cpu))

    for batch in range(batches):
        start_idx = batch * batch_size
        end_idx = min(start_idx + batch_size, total_data)
        
        # Select batch data
        x_batch = x[start_idx:end_idx]
        t_batch = t[start_idx:end_idx]
        
        # Compute model predictions and derivatives for the batch
        u_pred = model(x_batch, t_batch)
        # Now process residuals_batch
        for i in range(len(u_pred)):
            grad_outputs = torch.zeros_like(u_pred)
            grad_outputs[i] = 1
            # Compute gradients and immediately move them to CPU
            grads = [g.cpu() if g is not None else None for g in torch.autograd.grad(u_pred, param_tensors, grad_outputs, allow_unused=True, create_graph=True)]
            grad_row = torch.cat([g.view(-1) if g is not None else torch.zeros(p.numel()) for g, p in zip(grads, param_tensors_cpu)])
            jacobian_d[start_idx + i] = grad_row

    # Adapative weightings
    Krr = jacobian_r @ jacobian_r.T
    Kii = jacobian_i @ jacobian_i.T
    Kbb = jacobian_b @ jacobian_b.T
    Kdd = jacobian_d @ jacobian_d.T

    K_trace = torch.trace(Krr)/batch_sizes['PDE'] + torch.trace(Kii)/batch_sizes['initial'] +\
        torch.trace(Kbb)/batch_sizes['bounds'] + torch.trace(Kdd)/(batch_sizes['initial'] + batch_sizes['bounds'] + batch_sizes['data']) 
    epsilon = 1e-8  # Small regularization term
    lambda_r = K_trace / (torch.trace(Krr)/batch_sizes['PDE'] + epsilon)
    lambda_i = K_trace / (torch.trace(Kii)/batch_sizes['initial'] + epsilon)
    lambda_b = K_trace / (torch.trace(Kbb)/batch_sizes['bounds'] + epsilon)
    lambda_d = K_trace / (torch.trace(Kdd)/(batch_sizes['initial'] + batch_sizes['bounds'] + batch_sizes['data']) + epsilon) 

    lambda1 = (lambda_r.item())
    lambda2 = (lambda_i.item())
    lambda3 = (lambda_b.item())
    lambda4 = (lambda_d.item())

    return lambda1, lambda2, lambda3, lambda4

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
alpha_true = 1
batch_sizes = {'initial': 100, 'bounds': 200, 'PDE': 1000, 'data': 10000, 'validation': 10000}
epsilon_1 = 0.05
iter_1 = 200000 # Maximun number of iterations for Adam optimizer
iter_2 = 20000  # Maximun number  of iterations for L-BFGS optimizer
SNR = [40,35,30,25]
# =============================================================================
# TRAIN MODEL
# =============================================================================
for snr in SNR:
    torch.manual_seed(seeds_num)
    u_noisy, X, T, U, X_train, X_validation, X_test, X_true = get_data(alpha_true, batch_sizes, snr)
    epoch_loss_r = []
    epoch_loss_i = []
    epoch_loss_b = []
    epoch_loss_d = []
    epoch_alpha = []
    epoch_lambda1 = []
    epoch_lambda2 = []
    epoch_lambda3 = []
    epoch_lambda4 = []

    model = PINN(
        input_dim=2,
        output_dim=1,
        hidden_dim=100,
        num_hidden=3, 
        activation='sin'
    ).to(device)
    print(model)

    t11 = default_timer()
    # Adam optimizer to decrease loss in Phase 1
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
    train_dg_pinn(model, optimizer, X_train, iters=iter_1, stopping_loss=epsilon_1)

    lambda1, lambda2, lambda3, lambda4 = Adap_weights(model, X_train)

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

    u_pred = model(X_test['x'], X_test['t'])
        
    # Calculate relative L2 errors
    u_error = relative_l2_error(u_pred, X_test['u'])
    print(u_error.item())
    u_pred = u_pred.cpu().detach().numpy()    
    u_test = X_test['u'].cpu().detach().numpy()   

    U_pred = model(X_true['x'], X_true['t'])
    U_pred = U_pred.cpu().detach().numpy()    
    
    savemat(f'dgpinn_beam_NTK_SNR_{snr}.mat',
            {'u_pred': u_pred, 'u_test': u_test, 'U_pred': U_pred.reshape(201,201), 'u_noise': u_noisy.reshape(201,201), 'u_true': U, 'loss_r': epoch_loss_r, 'loss_i': epoch_loss_i,
             'loss_b': epoch_loss_b,'loss_d': epoch_loss_d, 'alpha': epoch_alpha, 'lambda1': epoch_lambda1, 
             'lambda2': epoch_lambda2, 'lambda3': epoch_lambda3, 'lambda4': epoch_lambda4, 'time': t22 - t11})     
  

  