import numpy as np
import cvxpy as cp
import mosek

import torch
import torch.nn
from torch import optim
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm as tqdm
from collections import OrderedDict
import math
import scipy.optimize as sopt

###########################################
# Algorithms for selecting sample weights
def uniform(sample_losses_history):
  num_samples = sample_losses_history.shape[1]
  a = np.ones((num_samples,), dtype=np.float) / num_samples
  return a

def adalpboost(sample_accuracy_history, sample_weights_history, eta):
  acc = sample_accuracy_history[-1]
  weight = sample_weights_history[-1]
  weight *= np.exp(-eta * acc)
  weight /= weight.sum()
  return weight

def adaboost(sample_accuracy_history, sample_weights_history):
  acc = sample_accuracy_history[-1]
  lss = 1 - acc
  weight = sample_weights_history[-1]
  eps = lss.T @ weight
  beta = eps / (1 - eps)
  weight *= np.power(beta, lss)
  weight /= weight.sum()
  return weight

############################
def find_opt_w(acc_history, P, eta, constraints_w, alpha=None, verbose=False): # max_{w in W} l(P, w) + eta * R(w)
  # assert abs(P.sum() - 1) < 1e-6 # P has to be a distribution
  l_all = 1 - acc_history
  num_epochs, n = l_all.shape
  w = cp.Variable(n)
  
  objective = cp.Maximize(P.T @ (l_all @ w) + eta * cp.sum(cp.entr(w)))
  constraints = []
  constraints.append(cp.sum(w) == 1)
  constraints.append(0 <= w)
  # add constraints
  nc = len(constraints_w)
  if 'cvar' in constraints_w:
    m = alpha * n
    constraints.append(w <= 1 / m)
    nc -= 1
  if 'chi2' in constraints_w:
    m = alpha * n
    constraints.append(cp.sum_squares(w) <= 1 / m)
    nc -= 1
  if 'none' in constraints_w:
    nc -= 1
  if nc > 0:
    print('Constraint {} not implemented'.format(constraints_w))
    raise NotImplementedError
  #################
  prob = cp.Problem(objective, constraints)
  result = lpsolver(prob, verbose=False)
  
  w_star = w.value
  w_star[w_star < 0] = 0
  w_star /= w_star.sum()
  return w_star, result
  
def find_opt_w_gdro(acc_history, P, eta, constraints_w, alpha=None, verbose=False, pop_masks=None): # max_{w in W} l(P, w) + eta * R(w)
  # assert abs(P.sum() - 1) < 1e-6 # P has to be a distribution
  l_all = 1 - acc_history
  num_epochs, n = l_all.shape
  m = len(pop_masks)
  sm = np.array([pop_masks[i].sum() for i in range(m)])
  w = cp.Variable(m)
  l_all_m = np.zeros((num_epochs, m))
  for i in range(m):
    l_all_m[:, i] = l_all[:, pop_masks[i]].sum(axis=1)
  if num_epochs >= 1:
    objective = cp.Maximize(P.T @ (l_all_m @ w) + eta * cp.sum(cp.multiply(sm, cp.entr(w))))
  else:
    objective = cp.Maximize(eta * cp.sum(cp.multiply(sm, cp.entr(w))))
  constraints = []
  constraints.append(cp.sum(cp.multiply(w, sm)) == 1)
  constraints.append(0 <= w)
  # add constraints
  nc = len(constraints_w)
  if 'cvar' in constraints_w:
    m = alpha * n
    constraints.append(w <= 1 / m)
    nc -= 1
  if 'chi2' in constraints_w:
    m = alpha * n
    constraints.append(cp.sum(cp.multiply(w ** 2, sm)) <= 1 / m)
    nc -= 1
  if 'none' in constraints_w:
    nc -= 1
  if nc > 0:
    print('Constraint {} not implemented'.format(constraints_w))
    raise NotImplementedError
  #################
  prob = cp.Problem(objective, constraints)
  result = lpsolver(prob, verbose=False)
  # print("RESULT", w.value)
  w_star = np.zeros((n,))
  for i in range(len(pop_masks)):
    w_star[pop_masks[i]] = w.value[i]
  w_star[w_star < 0] = 0
  w_star /= w_star.sum()
  return w_star, result

############################
def raigame(sample_accuracy_history, obj_value, eta_rai, constraints_w, hypothesis_weights=None, type=None, alpha=None,
            verbose=False, pop_masks=None):
  num_epochs, n = sample_accuracy_history.shape
  print("ETA_RAI", eta_rai)
  if type == 'gameplay':
    ans, result = find_opt_w(sample_accuracy_history, hypothesis_weights, eta_rai / (1 + num_epochs), constraints_w, alpha, verbose)
  elif type == 'greedy':
    ans, result = find_opt_w(sample_accuracy_history, hypothesis_weights, eta_rai, constraints_w, alpha, verbose)
  elif type == 'greedy_gdro':
    ans, result = find_opt_w_gdro(sample_accuracy_history, hypothesis_weights, eta_rai, constraints_w, alpha, verbose, pop_masks)
  else:
    print('Type {} not implemented'.format(type))
    raise NotImplementedError
  if obj_value is not None:
    obj_value.append(result)
  
  return ans

def get_hypothesis_weights(args, hypothesis_weights, sample_accuracy_history, pop_masks=None):
  t, n = sample_accuracy_history.shape
  assert hypothesis_weights.shape == (t - 1, 1)
  
  P = hypothesis_weights / hypothesis_weights.sum(axis=0)
  # do line search for a    
  base_a = 1 / t
  best_value = np.inf
  best_a = 0.25 * base_a
  for m_a in np.linspace(0.2, 2, 10): #[0.25, 0.5, 1, 2, 4]:
      a = base_a * m_a
      if a >= 1:
          continue
      P_temp = np.concatenate([(1 - a) * P, a * np.array([[1]])], axis=0)
      P_temp = P_temp / P_temp.sum()
      if pop_masks is not None:
        _, value = find_opt_w_gdro(sample_accuracy_history, P_temp, 0, args.constraints_w, args.alpha, verbose=False, pop_masks=pop_masks)
      else:
        _, value = find_opt_w(sample_accuracy_history, P_temp, 0, args.constraints_w, args.alpha, verbose=False)
      # assert value < np.inf
      if value < best_value:
          best_value = value
          best_a = a
  P = np.concatenate([(1 - best_a) * P, best_a * np.array([[1]])], axis=0)
  P = P / P.sum()
  return P

def get_hypothesis_weights_gen_adaboost(args, hypothesis_weights, sample_accuracy_history, pop_masks=None):
  t, n = sample_accuracy_history.shape
  assert hypothesis_weights.shape == (t - 1, 1)
  
  lamda = -1/2
  P = hypothesis_weights
  # do line search for a    
  base_a = 1
  best_value = np.inf
  best_a = None
  for m_a in np.linspace(0.1, 1.1, 20):
      a = base_a * m_a
      # if a >= 1:
      #     continue
      P_temp = np.concatenate([P, a * np.array([[1]])], axis=0)
      P_temp = P_temp / P_temp.sum()
      if pop_masks is not None:
        _, value = find_opt_w_gdro(sample_accuracy_history + lamda, P_temp, 0, args.constraints_w, args.alpha, verbose=False, pop_masks=pop_masks)
      else:
        _, value = find_opt_w(sample_accuracy_history + lamda, P_temp, 0, args.constraints_w, args.alpha, verbose=False)
      # assert value < np.inf
      if value < best_value:
          best_value = value 
          best_a = a
  P = np.concatenate([P, best_a * np.array([[1]])], axis=0)
  # P = P / P.sum()
  # print(P)
  return P

############################
# (Regularized) LPBoost
# Dual problem of LPBoost
def lpboost(sample_accuracy_history, obj_value, alpha,
            beta=None, verbose=False, solve_for_lbd=False):
  # Input shape: each history - (epoch, num_samples)
  # Output: group_weights  size: (num_samples,)
  num_epochs, n = sample_accuracy_history.shape

  w = cp.Variable(n)
  g = cp.Variable()
  objective = cp.Minimize(g) if beta is None else cp.Minimize(g - cp.sum(cp.entr(w)) / beta)
  constraints = [sample_accuracy_history[i, :] @ w <= g for i in range(num_epochs)]
  constraints.append(cp.sum(w) == 1)
  constraints.append(0 <= w)
  m = alpha * n
  constraints.append(w <= 1 / m)

  prob = cp.Problem(objective, constraints)
  result = lpsolver(prob, verbose)
  if obj_value is not None:
    obj_value.append(result)

  if solve_for_lbd:
    lbd = [constraints[i].dual_value for i in range(num_epochs)]
    lbd = np.array(lbd)
    lbd[lbd < 0] = 0
    lbd /= lbd.sum()
    return lbd

  ans = w.value
  ans[ans < 0] = 0
  ans /= ans.sum()
  return ans
   
############################
# Find optimal lambda (model weights)
# Solve the dual problem of LPBoost and use the values of primal variables
def find_opt_lbd(acc_history, alpha, verbose=False):
  # print('==> Computing optimal model weights...')
  lbd = lpboost(acc_history, None, alpha, None, verbose, True)
  # print('Optimal model weights found.')
  # print('Model weights: {}'.format(lbd))
  return lbd

###########################################
# Other functions
def test(model: Module, loader: DataLoader, criterion, device: str, label_id):
  """Test the avg and group acc of the model"""

  model.eval()
  total_correct = 0
  total_loss = 0
  total_num = 0
  l_rec = []
  c_rec = []

  with torch.no_grad():
    for _, (inputs, targets) in enumerate(loader):
      inputs, targets = inputs.to(device), targets.to(device)
      labels = targets if label_id is None else label_id(targets)
      outputs = model(inputs)
      predictions = torch.argmax(outputs, dim=1)
      c = (predictions == labels)
      c_rec.append(c.detach().cpu().numpy())
      correct = c.sum().item()
      l = criterion(outputs, labels).view(-1)
      l_rec.append(l.detach().cpu().numpy())
      loss = l.sum().item()
      total_correct += correct
      total_loss += loss
      total_num += len(inputs)

  print('Acc: {} ({} of {})'.format(total_correct / total_num, total_correct, total_num))
  print('Avg Loss: {}'.format(total_loss / total_num))

  l_vec = np.concatenate(l_rec)
  c_vec = np.concatenate(c_rec)

  return total_correct / total_num, total_loss / total_num, \
         c_vec, l_vec


def erm(model: Module, loader: DataLoader, optimizer: optim.Optimizer,
        criterion, scheduler, device: str, iters=0):
  """Empirical Risk Minimization (ERM)"""

  model.train()
  iteri = 0
  pbar = tqdm.tqdm(loader)
  average_loss = 0
  pbar.set_description('ERM')
  for _, (inputs, targets) in enumerate(pbar):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets).mean()
    #
    average_loss = (average_loss * iteri + loss.item()) / (iteri + 1)
    pbar_dic = OrderedDict()
    pbar_dic['batch_loss'] = loss.item()
    pbar_dic['avg_loss'] = average_loss
    pbar.set_postfix(pbar_dic)
    #
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    iteri += 1
    if iteri == iters:
      break
    if scheduler is not None:
      scheduler.step()

def chisq(model: Module, loader: DataLoader, optimizer: optim.Optimizer,
          criterion, scheduler, device: str, iters=0, alpha=0.95):
  """Chi^2-DRO"""

  model.train()
  max_l = 10.
  iteri = 0
  pbar = tqdm.tqdm(loader)
  average_loss = 0
  pbar.set_description('Chi^2-DRO')
  C = math.sqrt(1 + (1.0 / alpha - 1) ** 2)
  for _, (inputs, targets) in enumerate(pbar):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    foo = lambda eta: C * math.sqrt((F.relu(loss - eta) ** 2).mean().item()) + eta
    opt_eta = sopt.brent(foo, brack=(0, max_l))
    loss = C * torch.sqrt((F.relu(loss - opt_eta) ** 2).mean()) + opt_eta
    #
    average_loss = (average_loss * iteri + loss.item()) / (iteri + 1)
    pbar_dic = OrderedDict()
    pbar_dic['batch_loss'] = loss.item()
    pbar_dic['avg_loss'] = average_loss
    pbar.set_postfix(pbar_dic)
    #
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
      scheduler.step()
    iteri += 1
    if iteri == iters:
      break

domain_fn = [
        # lambda x: (x[:, 7] == 0) & (x[:, 6] == 1), # White and Female
        # lambda x: (x[:, 7] == 1) & (x[:, 6] == 0), # Other and Male
        # lambda x: (x[:, 7] == 0) & (x[:, 6] == 0), # White and Male
        # lambda x: (x[:, 6] == 1) & (x[:, 7] == 1), # Other and Female
        lambda x: (x[:, 7] == 0), # White
        lambda x: (x[:, 7] == 1), # Other
        ]

def group_dro_sagawa(model, grouploader, optimizer, criterion, scheduler, device, qs, iters=0, dec=0.01):
  """Group DRO (Sagawa et al., 2019)"""
  model.train()
  m = len(qs)
  iteri = 0
  pbar = tqdm.tqdm(grouploader)
  average_loss = 0
  pbar.set_description('GDRO')
  for _, (inputs, targets) in enumerate(pbar):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    for i in range(m):
      mask = (targets%m == i)
      # mask = domain_fn[i](inputs)
      if loss[mask].numel() == 0:
        # print('Warning: group {} has no data'.format(i))
        continue
      qs[i] = qs[i] * np.exp(loss[mask].mean().item() * dec)
    qs = qs / np.sum(qs)
    for i in range(m):
      mask = (targets%m == i)
      # mask = domain_fn[i](inputs)
      if loss[mask].numel() == 0:
        continue
      loss[mask] = loss[mask] * qs[i] * m
    optimizer.zero_grad()
    loss = loss.mean()
    loss.backward()
    # # update lr of optimizer
    # prev_lr = optimizer.param_groups[0]['lr']
    # for param_group in optimizer.param_groups:
    #   param_group['lr'] = prev_lr * qs[g] * m
    optimizer.step()
    # # restore lr of optimizer
    # for param_group in optimizer.param_groups:
    #   param_group['lr'] = prev_lr
    # #
    average_loss = (average_loss * iteri + loss.item()) / (iteri + 1)
    pbar_dic = OrderedDict()
    pbar_dic['batch_loss'] = loss.item()
    pbar_dic['avg_loss'] = average_loss
    pbar_dic['iteri'] = iteri
    pbar_dic['q_min'] = np.min(qs)
    pbar_dic['q_max'] = np.max(qs)
    pbar_dic['q_argmin'] = np.argmin(qs)
    pbar_dic['q_argmax'] = np.argmax(qs)
    pbar.set_postfix(pbar_dic)
    #
    if scheduler is not None:
      scheduler.step()
    iteri += 1
    if iteri == iters:
      break

############################
# LP Solver
# Try multiple solvers because single solver might fail
def lpsolver(prob, verbose=False):
  # print('=== LP Solver ===')
  solvers = [cp.MOSEK, cp.ECOS_BB]
  for s in solvers:
    # print('==> Invoking {}...'.format(s))
    try:
      result = prob.solve(solver=s, verbose=verbose)
      return result
    except cp.error.SolverError as e:
      print('==> Solver Error')

  # print('==> Invoking MOSEK simplex method...')
  try:
    result = prob.solve(solver=cp.MOSEK,
                      mosek_params={mosek.iparam.optimizer: mosek.optimizertype.free_simplex},
                      bfs=True, verbose=verbose)
    return result
  except cp.error.SolverError as e:
    print('==> Solver Error')

  raise cp.error.SolverError('All solvers failed.')