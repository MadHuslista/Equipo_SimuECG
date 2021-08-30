import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# Lectura de Argumentos externos 
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

# Elección de Adjoint or not. 
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# Traspaso a GPU de existir. 
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

#Vector de Tiempo para el ODE
t = torch.linspace(0., 25., args.data_size).to(device)

#Valores Iniciales del ODE
true_y0 = torch.tensor([[2., 0.]]).to(device)

# Parámetros reales del ODE 
    # Acá existen porque, para el ejemplo, es necesario
    # tenerlos para generar la señal de referencia. 
    # En la práctica son desconocidos y son los buscados por lo 
    # que aquí es el ODEFunc
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)



# This Lambda is the implementation of the real 
# differential equations. 
# Es decir, este Lambda es el que -para el ejemplo- 
# Crea la señal de referencia. 
# En la práctica no debería existir, y simplemente tendría la 
# señal ECG. 
class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)

# This is the functions to be trained. 
# It's parameters will be the ones that are trained. 
class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        # Esta es la net que se utilizará para entrenar los parámetros. 
        #self.net = nn.Sequential(
        #    nn.Linear(2, 50),
        #    nn.Tanh(),
        #    nn.Linear(50, 2),
        #)
        self.net = nn.Linear(2,2) # Lo dejé con una sola capa, porque de esta manera, la matriz de pesos sólo queda de 2x2, igual que True_A. 
            # Por tanto, como precisamente la matriz de peso con la de input, efectua una matmul (tal como en Lambda().forward() )
            # Al final del entrenamiento, la matriz de pesos debería llegar a valores similares de True_A

        #Este es el proceso de inicialización de los parámetros
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)


# El no_grad, implica que los cálculos efectuados por -este- odeint no se agregan 
# Al computational graph creado por el autograd, que después utiliza para hacer la bakcprop
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')

    #Este true_y, corresponde a las trayectorias de 'referencia', que en la práctica deberían corresponder a las señales ECG


def get_batch():
    # Determino posiciones aleatorias del vector de tiempo. 
    s = torch.from_numpy(
                        np.random.choice(
                                        np.arange(
                                                args.data_size - args.batch_time, 
                                                dtype=np.int64
                                                ), 
                                        args.batch_size, 
                                        replace=False
                                        )
                        )
    # Capturo los valores de true_y que corresponden a estas posiciones
    batch_y0 = true_y[s]  # (M, D)
    # Creo un sub-vector de tiempo según lo determinado. 
    batch_t = t[:args.batch_time]  # (T)

    # Luego, a partir de cada posición inicial random determinada, 
    # Capturo, desde true_y, tantos puntos como batch_time determine. 
    # En resumen: las señales de referencia, corresponden a segmentos aleatorios de true_y, de largo batch_time
    # Para la creación de uno de estos segmentos, la posición inicial se determina aleatoriamente mediante 'S'
    # y desde ahí se extrae el segmento de true_y que continua desde [s] durante [batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)

    # En dos palabras, estoy creando batch aleatorios, a través del muestreo segmentado de true_y
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)



# Estas funciones eran sólo para crear y guardar los png. 
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

# Visualización del entrenamiento
def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)



class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0

    #Creación de la función diferencial. Contiene los parámetros a buscar. 
    func = ODEFunc().to(device)
    
    # Observación de los parámetros originales. 
    #for params in func.parameters(): 
    #    print(params.shape)
    #    print(params)
    #    print()

    # Creación del optimizer. De manera clásica para DL. 
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
 
    #end = time.time()
    #time_meter = RunningAverageMeter(0.97)
    #loss_meter = RunningAverageMeter(0.97)

    # Loop de Entrenamiento. 
    for itr in range(1, args.niters + 1):
        
        # Zero Grad
        optimizer.zero_grad()

        # Obtención de los batch a partir del muestreo de true_y
        batch_y0, batch_t, batch_y = get_batch()

        # Obtención de las predicciones a partir del odeint en base a la func. 
        # En respuesta, me entrega las predicciones de la señal a partir de batch_y0. 
        # El ideal es que estas se correspondan con batch_y (que es la señal real, dado qeu es un segmento directo de true_y)
        pred_y = odeint(func, batch_y0, batch_t).to(device)

        # Cálculo de la pérdida, de manera normal.         
        loss = torch.mean(torch.abs(pred_y - batch_y))

        # Backpropagation
        loss.backward()

        #Step del optimizer. Aquí es donde se mueven los parámetros en dirección 
        # de los gradientes encontrados. 
        optimizer.step()

        #time_meter.update(time.time() - end)
        #loss_meter.update(loss.item())

        # Esta parte es sólo para la visualización. 
        if itr % args.test_freq == 0:
            with torch.no_grad():

                # Aquí se predice la señal completa, a partir de los valores iniciales originales 
                # Utilizando el estado actual de entrenamiento de los parámetros de la diferencial 
                pred_y = odeint(func, true_y0, t)

                # computo la pérdia, ahora entre la señal predicha completa, y el true_y (que es la ref completa)
                loss = torch.mean(torch.abs(pred_y - true_y))

                # Informo la pérdida
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))

                # Visualizo. 
                visualize(true_y, pred_y, func, ii)
                ii += 1

        #end = time.time()
    
    # Observación de los parámetros ya entrenados. 
    #for params in func.parameters(): 
    #    print(params.shape)
    #    #print(params.mean(), params.std())
    #    print(params)
    #    print()