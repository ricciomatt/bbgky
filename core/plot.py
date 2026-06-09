import copy
from qunum.jupyter_tools.plotting.engine import PlotIt, plotly_configs
from qunum.numerical import SUConnection, AdaptiveLinspace
from qunum.numerical import LazyTensor
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.core import rebuild_timeDep
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core import *
from qunum.numerical.mathematics.algebra import wigner_kernel
from qunum.numerical.physics.constants import h_barEvS as hbar
import torch, pickle, numpy as np, polars as pl, os, sys
from polars import col, int_ranges
from math import sqrt as msqrt
from qunum.numerical.physics.quantum.operators.dense.nuetrino import pmns2, pmns3
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.qi import oneBodyMagic, oneBodyMana, twoBodyMagic, twoBodyMana
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.qi import oneBodyRenyiEntropy
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.core import computeBaseMaps
from qunum.jupyter_tools import PlotIt 
import shutil

def pmnsrotPhi(Phi:torch.Tensor, sun:SUConnection|None = None)->torch.Tensor:
    print(sun.n)
    if(sun is None):
        sun = SUConnection(int(msqrt(Phi.shape[-1]+1)))
    Phi = torch.concat( (torch.ones((*Phi.shape[:-1],1), dtype =Phi.dtype)/sun.dR, Phi), dim = 2).to(torch.complex128)
    match sun.n:
        case 2:
            U = pmns2()
        case 3:
            U = pmns3()
    U = sun.decompose(U)
    
    return torch.einsum('u, ...v, x, uvp, pxk-> ...k', U.conj(), Phi, U, sun.g_uva, sun.g_uva)


def pmnsRotGamma(GammaBar:torch.Tensor, N:int, sun:SUConnection|None = None)->torch.Tensor:
    match sun.n:
        case 2:
            u = sun.decompose(pmns2())
        case 3:
            u = sun.decompose(pmns3())
    U = torch.einsum('uvp, pxy, u, x->vy', sun.g_uva, sun.g_uva, u.conj(), u)   
    Uf = pl.from_torch(torch.stack((torch.where(U.abs() > 1e-10 ))).T-1, 'uv').lazy()
    s = Uf.collect().shape[0]
    s-=1 
    Gmp:pl.LazyFrame = computeBaseMaps(N, sun.n, ret_dict= True)['Gmp']
    Gmp = Gmp = Gmp.join(
        Uf.rename({'v':'vA'}), left_on = 'a', right_on = 'u'
    ).join(
        Uf.rename({'v':'vB'}), left_on = 'a', right_on = 'u'
    ).join(
        Gmp.rename({'AaBb':'AvABvB',}), left_on = ('AB', 'vA','vB'), right_on = ('AB','a','b')
    ).group_by(
        pl.col('AvABvB'),pl.col('A'), pl.col('B'), pl.col('vA'),pl.col('vB') 
    ).agg(
        pl.col('AaBb'),pl.col('a'), pl.col('b')
    ).with_columns(pl.col('b').list.len().alias('l')).sort('AvABvB').collect()
    GT = torch.zeros_like(GammaBar).to(dtype = U.dtype)
    for l in Gmp.select('l').unique()['l']:
        T = {g.name:g.to_torch() for g in Gmp.filter(pl.col('l') == l).with_columns(pl.col('AaBb').list.to_array(l),pl.col('a').list.to_array(l), pl.col('b').list.to_array(l))}
        GT[:, T['AvABvB']] += (GammaBar[:, T['AaBb']]*U.view(1,*U.shape)[:, T['a']+1,T['vA'][:,None]+1]*U.view(1,*U.shape)[:, T['b']+1,T['vB'][:,None]+1]).sum(-1)
    return GT

def plot_flav(
        PhiBar:torch.Tensor, GammaBar:torch.Tensor, sun:SUConnection, plt:PlotIt, t:torch.Tensor, path_base:str = 'PaperImages',
        N:int = 100, 
        to_image:bool = True, show:bool=True,
        Nsamp:int = 200, **kwargs
    )->None|PlotIt:
    image_path = os.path.join(os.getcwd(), path_base, f'su{sun.n}', 'FlavorPolarization')
    os.makedirs(image_path, exist_ok = True)
    Pz = pmnsrotPhi(PhiBar, sun)[...,].real
    GT = pmnsRotGamma(GammaBar, N, sun).real
    
    plt.reset()
    color = ['rgba(255,0,0,.5)', 'rgba(0,0,255,.5)']
    color_Bar = ['rgb(0,0,0)', 'rgb(100,50,150)']
    dash = ['dot','dash']
    A = computeBaseMaps(N, sun.n, ret_dict=True)['Gmp']
    
    for m, k in enumerate((torch.arange(2,sun.n+1).pow(2)-1)):
        plt.extend_data(*(
            dict(
                x = t, y = Pz[...,i,k].real, 
                marker=dict(color = color[m]), line = dict(width = 5), 
                showlegend = bool(i<1), 
                name = r'$\Huge\Phi^{\text{Flavor}}_{A'+str(int(k))+'}$',legend = 'legend',
            ) for i in range(N)
        ))
        plt.add_data(
            x = t, y = Pz[...,k].real.mean(-1), 
            marker=dict(color = color_Bar[m]), line = dict(width = 10, dash = dash[m]), 
            showlegend = True, name = r'$\Huge\bar{\Phi}^{\text{Flavor}}_{'+str(int(k))+'}$',legend = 'legend',
        )

    
    
    
    plt.update_layout(
        xaxis = dict(title = dict(text = r'$\Huge {t(\omega_{0}^{-1})}$'), tickfont = dict(size = 35)),
        yaxis = dict(title = dict(text = r"$\Huge \Phi_{A\mathfrak{d}}^{\text{Flavor}}$"), tickfont = dict(size = 35)),
        legend  = dict(title = dict(text = r''), font = dict(size = 28), xanchor = 'left', itemwidth=85, bordercolor="black", borderwidth=3),
        
        title = dict(text = r"Flavor Population Evolution", font = dict(size = 50, weight = 'bold', family = 'Times New Roman')),
    )
    plt.update_layout(**kwargs)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'OneBodyFlavor.png'))
    
    plt.reset()
    print(Pz.shape)
    for m, k in enumerate((torch.arange(2,sun.n+1).pow(2)-1)):
        IX = A.filter(pl.col('a').__eq__(k-1) & pl.col('b').__eq__(k-1)).collect().sample(Nsamp)['AaBb', 'a', 'b', 'A','B']
        
        plt.extend_data(*(
            dict(
                x = t, y = GT[...,i['AaBb']].real - Pz[:, i['A'],k]*Pz[:,i['B'], k], 
                marker=dict(color = color[m]), line = dict(width = 5), 
                showlegend = bool(j<1), xaxis = 'x', yaxis = 'y',
                name =r'$\Huge\Sigma^{\text{Flavor}}_{A'+str(int(k))+'B'+str(int(k))+'}$' ,legend = 'legend',
            ) for j,i in enumerate(IX.to_dicts())
        ))
        IX =  A.filter(pl.col('a').__eq__(k-1) & pl.col('b').__eq__(k-1)).collect()
        Sigma = GT[:,IX['AaBb']].real - Pz[:, IX['A'],k] * Pz[:, IX['B'], k]
        plt.add_data(
            x = t, y = Sigma.mean(-1) , 
            marker=dict(color = color_Bar[m]), line = dict(width = 10, dash = dash[m]), xaxis = 'x', yaxis = 'y',legend = 'legend',
            showlegend = True, name = r'$\Huge\bar{\Sigma}^{\text{Flavor}}_{'+str(int(k))+''+str(int(k))+'}$'
        )
        print(Sigma[0])
    
    plt.update_layout(
        xaxis = dict(title = dict(text = r'$\Huge {t(\omega_{0}^{-1})}$'), tickfont = dict(size = 35)),
        yaxis = dict(title = dict(text = r"$\Huge \Sigma_{A\mathfrak{d}B\mathfrak{d}}^{\text{Flavor}}$"), tickfont = dict(size = 35)),
        legend  = dict(title = dict(text = r''), font = dict(size = 28), xanchor = 'left', itemwidth=85, bordercolor="black", borderwidth=3),
        title = dict(text = r"Second Cumulant for Flavor Population Evolution", font = dict(size = 50, weight = 'bold', family = 'Times New Roman')),
      
    )
    plt.update_layout(**kwargs)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'TwoBodyFlavorCum.png'),)
    return 

def plot_ptrans(
    PhiBar:torch.Tensor, tfact:float, t:torch.Tensor, 
    sun:SUConnection, B:LazyTensor, u:LazyTensor, 
    plt:PlotIt|None = None,
    path_base:str = 'PaperImages',N:int = 100, 
    to_image:bool = True, show:bool=True,
    Nsamp:int = 200, **kwargs
)->None|PlotIt:
    image_path = os.path.join(os.getcwd(), path_base, f'su{sun.n}', 'PhaseTransition')
    os.makedirs(image_path, exist_ok = True)
    if(plt is None):
        plt = PlotIt()
    else:
        plt.reset()
    plt.subplot_grid(ncols=1, nrows=2, spacingr=.1)

    #S = oneBodyRenyiEntropy(PhiBar, sun = sun, )

    #plt.extend_data(*(
    #        dict(
    #            x = t, y =  S[:,i],
    #            xaxis = 'x', yaxis = 'y', legend = 'legend',
    #            line = dict(width = 10, dash = 'solid'), marker = dict(color = 'rgba(255,0,0,.25)',),
    #            showlegend = bool(i == 0), name = '$\\Huge S_{2,A}$',legendgroup = 'S', 
    #        ) for i in range(N )
    #    )
    #)
    #St = S.std(-1)
    #S = S[:,].mean(-1)
    #plt.add_data(
    #    x = t, y=  S,
    #    xaxis = 'x', yaxis = 'y', legend = 'legend',
    #    line = dict(width = 7.5, dash='solid'), marker=dict(color='black'), 
    #
    #    showlegend = True, name = '$\\Huge \\bar{S}_{2}$', legendgroup = 'S'
    #)
    #plt.add_data(
    #    x = t, y=  S+St,
    #    xaxis = 'x', yaxis = 'y', legend = 'legend',
    #    line = dict(width = 7.5, dash='dash'), marker=dict(color='black'),
    #    showlegend = True, name = r'$\Huge \bar{S}_{2} \pm \sigma_{(S_{2})}$', legendgroup = 'S'
    #)
    #plt.add_data(
    #    x = t, y=  S-St,
    #    xaxis = 'x', yaxis = 'y', legend = 'legend',
    #    line = dict(width = 7.5, dash='dash'), marker=dict(color='black'),
    #    showlegend = False, legendgroup = 'S'
    #)
    
    plt.add_data(
        x = t, y = PhiBar.real.mean(1).pow(2).sum(-1).real, 
        marker = dict(color = 'blue'), line = dict(width = 4, dash = 'solid'),
        yaxis = 'y', xaxis = 'x',  legend = 'legend',
        name = '$\\Huge \\bar{\\Phi}_a\\bar{\\Phi}^a$', legendgroup = 'A', showlegend = True, 
    )
    
    plt.add_data(
        x = [0, t[-1]], y = [B(0)[:,2].real.mean().item(), B(0)[:,2].real.mean().item()],
        line=dict(width = 4, dash='dash'), marker=dict(color = 'orange'),   mode='lines',
        yaxis = 'y2', xaxis ='x2', legend = 'legend2', 
        name = '$\\Huge|\\bar{\\mathbf{B}}|$', showlegend = True
    )
    plt.add_data(
        x = t, y = u(tfact*t).real, 
        yaxis= 'y2', xaxis = 'x2', legend = 'legend2',
        line=dict(width = 4), marker = dict(color = 'purple'), 
        name = '$\\Huge \\mu_{(t)}$', showlegend = True
    )
   
    
    titles = {
        '2':'Energy  Scales', 
        '':'Order  Parameter', 
        } 
    annotations = [
        dict(
            text=val,
            xref='paper',
            yref='paper',
            font = dict(size= 50, family = 'Times New Roman', weight = 'bold'),
            x = plt.layout[f'xaxis{key}']['domain'][0]+.01,
            y = plt.layout[f'yaxis{key}']['domain'][1]+.015,
            xanchor = 'left',
            yanchor = 'top',
        )
        for key, val in titles.items()
    ]
    
    ix = (((u(t[:-1]*tfact))- (B(0)[:,2].real.mean().item())).abs()<1e-3)
    time = (t[:-1][ix][0])
    
    shapes = [
        dict(
        type= 'line', xref= f'x{tit}', yref= 'paper', 
        x0 = time, 
        x1 = time,
        y0 =plt.layout[f'yaxis{tit}']['domain'][0], y1 = plt.layout[f'yaxis{tit}']['domain'][1], line = dict(color = 'rgb(255,0,0)', width = 5, dash = 'dot'), 
        name = r'$\Huge |\bar{\mathbf{B}}|\sim \mu_{(t)}$', legend = 'legend2', showlegend = False
        )
        for tit in titles
    ]
    plt.update_layout(
        height = 5000/6*3, width = 2000, 
        title = dict(
            text = r'$\Huge\textbf{Dynamical  Phase  Transition around}(\mu_{(t)}\sim |\mathbf{B}|)$',
            font = dict(size = 60, weight = 'bold', family='Times New Roman'),
            yref= 'paper', x=0, xref = 'paper',
        ),
        yaxis= dict(title = dict(text = r"$\Huge|\bar{\mathbf{\Phi}}|^2$"), tickfont=dict(size = 35)),
        yaxis2= dict(title = dict(text = r"$\Huge \mu_{(t)}$"), tickfont=dict(size = 35)),
        
        xaxis = dict(title = dict(text = r"$\Huge t(\omega_{0}^{-1})$"), tickfont=dict(size = 35)),
        xaxis2 = dict(title = dict(text = r"$\Huge t(\omega_{0}^{-1})$"), tickfont=dict(size = 35)),
        
        legend  = dict( font = dict(size = 28), xanchor = 'left', itemwidth=85, bordercolor="black", borderwidth=3),
        legend2 = dict(font = dict(size = 28), xanchor = 'left', itemwidth=110, bordercolor="black", borderwidth=3),
        
        shapes = shapes,
        annotations = annotations
    
        
    )
    plt.update_layout(**kwargs)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'PhaseTransition.png'))

def plot_magic_mana(
    PhiBar:torch.Tensor, GammaBar:torch.Tensor,
    tfact:float, t:torch.Tensor, sun:SUConnection,
    plt:PlotIt|None = None,
    path_base:str = 'PaperImages',N:int = 100, 
    to_image:bool = True, show:bool=True,
    Nsamp:int = 200, **kwargs
)->None:
    image_path = os.path.join(os.getcwd(), path_base, f'su{sun.n}', 'MagicAndMana')
    os.makedirs(image_path, exist_ok = True)
    if(plt is None):
        plt = PlotIt()
    else:
        plt.reset()
    # one Body
    M2 = oneBodyMagic(PhiBar.clone(),)
    plt.extend_data(*(
            dict(
                x = t, y =  M2[:,i],
                xaxis = 'x', yaxis = 'y', legend = 'legend',
                line = dict(width = 10, dash = 'solid'), marker = dict(color = 'rgba(255,0,0,.25)',),
                showlegend = bool(i == 0), name = '$\\Huge \\mathscr{M}_{2,A}$', 
                
            ) for i in range(N)
        )
    )
    plt.add_data(
        x = t, y=  M2[:,].mean(-1),
        xaxis = 'x', yaxis = 'y', legend = 'legend',
        line = dict(width = 7.5, dash='dot'), marker=dict(color='black'),
        showlegend = True, name = '$\\Huge \\bar{\\mathscr{M}}_{2,A}$', 
    )
    plt.update_layout(
        title = dict(
            text = r'One-Body Magic', font=dict(weight = 'bold', size = 50, family = 'Times New Roman'),
            yref= 'paper', x=0, xref = 'paper'
        ),
        yaxis= dict(title = dict(text = r"$\Huge \mathscr{M}_{2A}$"), tickfont=dict(size = 35)),
        xaxis = dict(title = dict(text = r"$\Huge t(\omega_{0}^{-1})$"), tickfont= dict(size= 35)),
        legend  = dict(title = dict(text = r""), font = dict(size = 28), xanchor = 'left', itemwidth=85, bordercolor="black", borderwidth=3),
        **kwargs,
    )
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'OneBodyMagic.png'))
    
    plt.reset()    
    M2 = oneBodyMana(PhiBar.clone(),)
    plt.extend_data(*(
            dict(
                x = t, y =  M2[:,i],
                xaxis = 'x', yaxis = 'y', legend = 'legend',
                line = dict(width = 10, dash = 'solid'), marker = dict(color = 'rgba(255,0,0,.25)',),
                showlegend = bool(i == 0), name = '$\\Huge \\mathcal{M}_{A}$', 
                
            ) for i in range(N )
        )
    )
    plt.add_data(
        x = t, y=  M2[:,].mean(-1),
        xaxis = 'x', yaxis = 'y', legend = 'legend',
        line = dict(width = 7.5, dash='dot'), marker=dict(color='black'),
        showlegend = True, name = '$\\Huge \\bar{\\mathcal{M}}_{A}$'
    )

    plt.update_layout(
        title = dict(
            text = r'One-Body Mana', font=dict(weight = 'bold', size = 50, family = 'Times New Roman'),
            yref= 'paper', x=0, xref = 'paper'
        ),
        yaxis= dict(title = dict(text = r"$\Huge \mathcal{M}_{A}$"), tickfont=dict(size = 35)),
        xaxis = dict(title = dict(text = r"$\Huge t(\omega_{0}^{-1})$"), tickfont= dict(size= 35)),
        legend  = dict(title = dict(text = r""), font = dict(size = 28), xanchor = 'left', itemwidth=85, bordercolor="black", borderwidth=3),
        **kwargs
    )

    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'OneBodyMana.png'))
    plt.reset()

   #two Body
    print('Computing two body Magic')
    M2 = twoBodyMagic(PhiBar.clone(), GammaBar.clone(), sun, ret_map=False)
    print('Computed two body Magic')
    IX = torch.randint(0, N*(N-1)//2, (Nsamp,))
    plt.extend_data(*(
            dict(
                x = t, y =  M2[:,i].real,
                xaxis = 'x', yaxis = 'y', legend = 'legend',
                line = dict(width = 10, dash = 'solid'), marker = dict(color = 'rgba(255,0,0,.25)',),
                showlegend = bool(j == 0), name = '$\\Huge \\mathscr{M}_{2,AB}$', 
                
            ) for j,i in enumerate(IX)
        )
    )
    plt.add_data(
        x = t, y=  M2[:,].mean(-1).real,
        xaxis = 'x', yaxis = 'y', legend = 'legend',
        line = dict(width = 7.5, dash='dot'), marker=dict(color='black'),
        showlegend = True, name = '$\\Huge \\bar{\\mathscr{M}}_{2,AB}$', 
    )

    plt.update_layout(
        title = dict(
            text = r'Two-Body Magic', font=dict(weight = 'bold', size = 50, family = 'Times New Roman'),
            yref= 'paper', x=0, xref = 'paper'
        ),
        yaxis= dict(title = dict(text = r"$\Huge \mathscr{M}_{2,AB}$"), tickfont=dict(size = 35)),
        xaxis = dict(title = dict(text = r"$\Huge t(\omega_{0}^{-1})$"), tickfont= dict(size= 35)),
        legend  = dict(title = dict(text = r""), font = dict(size = 28), xanchor = 'left', itemwidth=85, bordercolor="black", borderwidth=3),
        **kwargs
    )

    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'TwoBodyMagic.png'))
    plt.reset()

    
    print('Computing two body Magic')
    M2, _ = twoBodyMana(PhiBar.clone(), GammaBar.clone(), sun, ret_map=True)
    print('Computed two body Magic')
    plt.extend_data(*(
            dict(
                x = t, y =  M2[:,i].real,
                xaxis = 'x', yaxis = 'y', legend = 'legend',
                line = dict(width = 10, dash = 'solid'), marker = dict(color = 'rgba(255,0,0,.25)',),
                showlegend = bool(j == 0), name = '$\\Huge \\mathcal{M}_{AB}$', 
                
            ) for j,i in enumerate(IX)
        )
    )
    plt.add_data(
        x = t, y=  M2[:,].mean(-1).real,
        xaxis = 'x', yaxis = 'y', legend = 'legend',
        line = dict(width = 7.5, dash='dot'), marker=dict(color='black'),
        showlegend = True, name = '$\\Huge \\bar{\\mathcal{M}}_{AB}$'
    )
    plt.update_layout(
        title = dict(
            text = r'Two-Body Mana', font=dict(weight = 'bold', size = 50, family = 'Times New Roman'),
            yref= 'paper', x=0, xref = 'paper'
        ),
        yaxis= dict(title = dict(text = r"$\Huge \mathcal{M}_{AB}$"), tickfont=dict(size = 35)),
        xaxis = dict(title = dict(text = r"$\Huge t(\omega_{0}^{-1})$"), tickfont= dict(size= 35)),
        legend  = dict(title = dict(text = r""), font = dict(size = 28), xanchor = 'left', itemwidth=85, bordercolor="black", borderwidth=3),
        **kwargs
    )

    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'TwoBodyMana.png'))
    
    return 

def plot_non_local(
    PhiBar:torch.Tensor, GammaBar:torch.Tensor, t:torch.Tensor, pMag:torch.Tensor, 
    sun:SUConnection, lfs:dict[str:pl.LazyFrame],
    plt:PlotIt|None = None, path_base:str = 'PaperImages', N:int = 100, 
    to_image:bool = True, show:bool=True,
    Nsamp:int = 200, height_hmap:int = 1500, width_hmap:int = 1500,
    **kwargs
)->None|PlotIt:
    image_path = os.path.join(os.getcwd(), path_base, f'su{sun.n}', 'NonLocal')
    os.makedirs(image_path, exist_ok = True)
    if(plt is None):
        plt = PlotIt()
    else:
        plt.reset()

    
    lf:pl.LazyFrame = lfs['lf'].clone()
    IXS = lf.select(
        'A','B','C'
    ).explode(
        'A'
    ).with_columns(
        col('B').arr.to_list().list.set_intersection(int_ranges(col('A')+1, N))
    ).filter(
        col('B').list.len().__eq__(0).not_()
    ).explode(
        'B'
    ).with_columns(
        col('C').arr.to_list().list.set_intersection(int_ranges(col('B')+1, N))
    ).filter(
        col('C').list.len().__eq__(0).not_()
    ).explode(
        'C'
    ).collect().sample(Nsamp).sort('A','B','C')
    IXS = IXS.to_numpy()
    tI, Mp = mutualInformation(PhiBar, GammaBar, sun =sun, lfs = lfs, ret_map= True)
    I = torch.zeros((tI.shape[0], PhiBar.shape[1], PhiBar.shape[1]), dtype = tI.dtype)
    I[:, Mp['Aix'], Mp['Bix']] = tI[:, Mp['ABix']]
    print('I2')
    S2, Mp2 = twoBodyRenyiEntropy(PhiBar, GammaBar,ret_map=True, retfull=True)
   
    S = torch.zeros((tI.shape[0], PhiBar.shape[1], PhiBar.shape[1]), dtype = tI.dtype)
    S[:, Mp['Aix'], Mp['Bix']] = S2[:, Mp['AB']]
    print('S2')
    I3 = list(
        (
            twoBodyRenyiEntropy(PhiBar, GammaBar, A = np.sort(ix[:2]), B = [ix[2]]).sum(-1) - oneBodyRenyiEntropy(PhiBar, A = ix[2]).flatten() - threeBodyRenyiEntropy(PhiBar, GammaBar,A = ix[0], B=ix[1], C=ix[2] ).flatten()
            for ix in IXS
        )
    ) 
    print('I3')


    #3-Body Renyi
    S3 = list(threeBodyRenyiEntropy(PhiBar, GammaBar, sun, lfs, A = ix[0], B = ix[1], C=ix[2]).real for ix in IXS)
    
    S3Bar = sum(S3)/Nsamp

    print('S3')

    print('Computed')
    


    
    #3-Body Entropy
    plt.reset()
    plt.extend_data(*(dict(x = t, y = S3[i], name = r'$\Huge S_{2,ABC}$', marker = dict(color = 'rgba(255,0,0,.5)',), line = dict(width = 7.5), showlegend = bool(i==0)) for i in range(Nsamp)))
    plt.add_data(x = t, y = S3Bar, name = r'$\Huge \bar{S}_{2,ABC}$', marker = dict(color = 'rgba(0,0,0,1)',), line = dict(width = 7.5, dash = 'dot'))
    plt.update_layout(
        xaxis = dict(title = '$\\Huge t(\\omega_{0}^{-1})$', tickfont = dict(size = 35)),
        yaxis =  dict(title = r'$\Huge S_{2,ABC} $',tickfont = dict(size = 35)),
        title = dict(text = r'Three Body Entropy', font = dict(size = 50, family = 'Times New Roman', weight = 'bold'))
    )
    plt.update_layout(**kwargs)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'ThreeBodyEntorpy.png'))
    

    #3-body Mutual Infor

    plt.reset()
    plt.extend_data(*(dict(x = t, y = I3[i].real, name = r'$\Huge \mathcal{I}_{2,(A:B:C)}$', marker = dict(color = 'rgba(255,0,0,.5)',), line = dict(width = 7.5), showlegend = bool(i==0)) for i in range(Nsamp)))
    plt.add_data(x = t, y = sum(I3).real/len(I3), name = r'$\Huge \bar{\mathcal{I}}_{2,(A:B:C)}$', marker = dict(color = 'rgba(0,0,0,1)',), line = dict(width = 7.5, dash = 'dot'))
    plt.update_layout(
        xaxis = dict(title = '$\\Huge t(\\omega_{0}^{-1})$', tickfont = dict(size = 35)),
        yaxis =  dict(title = r'$\Huge \mathcal{I}_{2,(A:B:C)} $',tickfont = dict(size = 35)),
        title = dict(text = r'Three-Body Mutual Information Estimator', font = dict(size = 50, family = 'Times New Roman', weight = 'bold'))
    )
    plt.update_layout(**kwargs)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'ThreeBodyMutInfo.png'))


    #2-body Mutual Info
    pIx = np.random.choice(np.arange(0, tI.shape[1]), (Nsamp), replace = False)
    plt.reset()
    plt.extend_data(*(dict(x = t, y = tI[:,i].real, name = r'$\Huge \mathcal{I}_{2,(A:B)}$', marker = dict(color = 'rgba(255,0,0,.5)',), line = dict(width = 7.5), showlegend = bool(j==0)) for j,i in enumerate(pIx)))
    plt.add_data(x = t, y = tI.mean(-1).real, name = r'$\Huge \bar{\mathcal{I}}_{2,(A:B)}$', marker = dict(color = 'rgba(0,0,0,1)',), line = dict(width = 7.5, dash = 'dot'))
    plt.update_layout(
        xaxis = dict(title = '$\\Huge t(\\omega_{0}^{-1})$', tickfont = dict(size = 35)),
        yaxis =  dict(title = r'$\Huge \mathcal{I}_{2,(A:B)} $',tickfont = dict(size = 35)),
        title = dict(text = r'Two-Body Mutual Information Estimator', font = dict(size = 50, family = 'Times New Roman', weight = 'bold'))
    )
    plt.update_layout(**kwargs)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'TwoBodyMutInfo.png'))


    
    #2-body Mutual Info Heat Map
    i = -1
    plt.reset()
    plt.add_data(
        type = 'heatmap',
        x = pMag.real, y = pMag.real, z = I[i].real + I[i].T.real, 
        zmin = float(I.real.min().real), zmax =  float(I.real.max().real),
        colorscale='rdbu_r', 
    )
    annotation = dict(
            text=r'$\Huge\mathcal{I}_{2,(A:B)}$',
            xref='paper',
            yref='paper',
            font = dict(size= 35),
            x = 1.01,
            y = 1.01,
            xanchor = 'left',
            yanchor = 'top',
    )
    plt.update_layout(
        xaxis = dict(title = r'$\Huge \frac{|\mathbf{p}|}{E_0}$', tickfont = dict(size = 35)),
        yaxis =  dict(title = r'$\Huge \frac{|\mathbf{p}|}{E_0}$',tickfont = dict(size = 35)),
        title = dict(text = r'Two-Body Mutual Information Estimator', font = dict(size = 50, family = 'Times New Roman', weight = 'bold')),
        annotations = [annotation]
    )
    plt.update_layout(height = height_hmap, width =width_hmap)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'TwoBodyMutInfoHMap.png'))


    #2-body Entropy Heat Map
    plt.reset()
    i = -1
    plt.add_data(
        x = pMag.real, y = pMag.real, z = S[i].real + S[i].real.T, 
        type = 'heatmap', 
        zmin = float(S.real.min()), zmax = float(S.real.max()), 
        colorscale='rdbu_r',
    )
    annotation = dict(
            text=r'$\Huge S_{2,AB}$',
            xref='paper',
            yref='paper',
            font = dict(size= 35),
            x = 1.01,
            y = 1.01,
            xanchor = 'left',
            yanchor = 'top',
    )
    plt.update_layout(
        xaxis = dict(title = r'$\Huge \frac{|\mathbf{p}|}{E_0}$', tickfont = dict(size = 35)),
        yaxis =  dict(title = r'$\Huge \frac{|\mathbf{p}|}{E_0}$',tickfont = dict(size = 35)),
        title = dict(text = r'Two-Body Entropy', font = dict(size = 50, family = 'Times New Roman', weight = 'bold')),
        annotations = [annotation]
        
    )


    plt.update_layout(height = height_hmap, width =width_hmap)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'TwoBodyEntHMap.png'))



    #2-body Mutual Info
    plt.reset()
    plt.extend_data(*(dict(x = t, y = S2[:,i], name = r'$\Huge S_{2,AB}$', marker = dict(color = 'rgba(255,0,0,.5)',), line = dict(width = 7.5), showlegend = bool(j==0)) for j,i in enumerate(pIx)))
    plt.add_data(x = t, y = S2.mean(-1), name = r'$\Huge S_{2,(A:B)}$', marker = dict(color = 'rgba(0,0,0,1)',), line = dict(width = 7.5, dash = 'dot'))
    plt.update_layout(
        xaxis = dict(title = r'$\Huge t(\omega_{0}^{-1})$', tickfont = dict(size = 35)),
        yaxis =  dict(title = r'$\Huge S_{2,AB} $',tickfont = dict(size = 35)),
        title = dict(text = r'Two-Body Entropy', font = dict(size = 50, family = 'Times New Roman', weight = 'bold'))
    )
    plt.update_layout(**kwargs)
    if(show):
        plt.show(renderer = 'png')
    if(to_image):
        plt.to_image(os.path.join(image_path, 'TwoBodyEnt.png'))


   
    return



