import copy
from qunum.jupyter_tools.plotting import PlotIt
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
from qunum.numerical.physics.quantum.heisenberg.bbgky_truncation.core.core import pmnsrotPhi

def plot_non_local(
        PhiBar:torch.Tensor, GammaBar:torch.Tensor, t:torch.Tensor, pMag:torch.Tensor, 
        plot_path:str, sun:SUConnection, lfs:dict[str:pl.LazyFrame], plt:PlotIt, to_image:bool = True,
    )->None:
    NSamp = 200
    N = 100
    lf:pl.LazyFrame = lfs['lf']
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
    ).collect().sample(NSamp).sort('A','B','C')

    IXS = IXS.to_numpy()
    tI, Mp = mutualInformation(PhiBar, GammaBar, sun =sun, lfs = lfs, ret_map= True)
    I = torch.zeros((tI.shape[0], PhiBar.shape[1], PhiBar.shape[1]), dtype = tI.dtype)
    I[:, Mp['Aix'], Mp['Bix']] = tI[:, Mp['ABix']]

    S2, Mp2 = twoBodyRenyiEntropy(PhiBar, GammaBar,ret_map=True, retfull=True)
    print(Mp)
    S = torch.zeros((tI.shape[0], PhiBar.shape[1], PhiBar.shape[1]), dtype = tI.dtype)
    S[:, Mp['Aix'], Mp['Bix']] = S2[:, Mp['AB']]
    I3 = list(
        (
            twoBodyRenyiEntropy(PhiBar, GammaBar, A = np.sort(ix[:2]), B = [ix[2]]).sum(-1) - oneBodyRenyiEntropy(PhiBar, A = ix[2]).flatten() - threeBodyRenyiEntropy(PhiBar, GammaBar,A = ix[0], B=ix[1], C=ix[2] ).flatten()
            for ix in IXS
        )
    )


    #3-Body Renyi
    S3 = list(threeBodyRenyiEntropy(PhiBar, GammaBar, sun, lfs, A = ix[0], B = ix[1], C=ix[2]).real for ix in IXS)
    S3Bar = sum(S3)/NSamp



    print('Computed')


    NSamp2 = 300
    plt.reset()
    plt.subplot_grid(ncols=1, nrows=4, spacingr=.05, leg_opts=dict(bordercolor="black", borderwidth=1))

    #2 Body Mudual Information
    plt.extend_data(
        *(
            dict(
                x = t, y = tI[:,i], 
                name = '$\\huge\\mathcal{I}_{2,AB}$', showlegend = bool(j == 0),
                line = dict(width = 7.5), marker = dict(color = 'rgba(255, 0,0,.25)'),
                xaxis = 'x', yaxis = 'y', legend = 'legend',
            ) 
            for j,i in enumerate(torch.randint(0, tI.shape[1], (NSamp2,)))
        )
    )


    plt.add_data(
        x = t, y = tI.mean(-1), 
        name = '$\\huge\\bar{\\mathcal{I}}_{2,AB}$', 
        marker=dict(color='black'), line =dict(width = 7.5),
        xaxis = 'x', yaxis = 'y', legend = 'legend',
    )
    i = -1




    #2body-HeatMap
    dm = plt.layout['yaxis2']['domain']
    plt.add_data(
        type = 'heatmap',
        x = pMag.real, y = pMag.real, z = I[i] + I[i].T, 
        xaxis = 'x2', yaxis = 'y2', legend='legend2',
        zmin = 0, zmax = .15, 
        colorscale='rdbu_r', 
        colorbar = dict(yref = 'paper', xref='paper', yanchor = 'top', x = .45, y=dm[-1], len = dm[-1]-dm[0]), 
    )

    #2-Body Entropy

    plt.extend_data(
        *(
            dict(
                x = t, y = S2[:,i], 
                xaxis = 'x5', yaxis='y5', legend = 'legend5',
                name = r'$\huge S_{2,AB}$', showlegend = bool(j == 0),
                  line = dict(width = 7.5), marker = dict(color = 'rgba(255, 0,0,.25)')
                ) for j,i in enumerate(torch.randint(0, S2.shape[1], (NSamp2,)))
        )
    )
    plt.add_data(
        x = t, y = S2.mean(-1), 
        xaxis = 'x5', yaxis='y5', legend = 'legend5',
        name = r'$\huge\bar{S}_{2,AB}$',
        marker=dict(color='black'), line =dict(width = 7.5) 
    )

    dm = plt.layout['yaxis2']['domain']
    plt.add_data(
        x = pMag.real, y = pMag.real, z = S[i] + S[i].T, 
        type = 'heatmap', xaxis = 'x6', yaxis = 'y6', legend='legend6', 
        zmin = 0, zmax = .45, 
        colorscale='rdbu_r',
        colorbar = dict(yref = 'paper', xref='paper', yanchor = 'top', x = 1, y=dm[-1], len = dm[-1]-dm[0]), 
    )



    #3-Body Mutual-Information

    plt.extend_data(
        *(
            dict(
                x = t, y = I3[i].real, showlegend = bool(i<1), 
                xaxis = 'x3', yaxis = 'y3', legend='legend3', 
                name = '$\\huge \\mathcal{I}(AB:C)$', 
                line = dict(width = 15), marker = dict(color = 'rgba(255, 0,0,.25)')
            ) 
            for i,ix in enumerate(IXS)
        )
    )
    plt.add_data(
        x = t, y = sum(I3).real/len(I3), 
        name = '$\\huge \\bar{\\mathcal{I}}(AB:C)$', 
        xaxis = 'x3', yaxis = 'y3', legend = 'legend3', 
        line = dict(width = 7.5), marker = dict(color = 'rgb(0,0,0)')
    )

    plt.extend_data(
        *(
            dict(
                x = t, y = S3[i], 
                showlegend = bool(i<1), name = '$\\huge S_{2,ABC}$',
                  xaxis = 'x4', yaxis = 'y4', legend='legend4', 
                  line = dict(width = 15), marker = dict(color = 'rgba(255, 0,0,.25)')
            ) 
            for i,ix in enumerate(IXS)
        )
    )
    plt.add_data(
        x = t, y = S3Bar, 
        name = '$\\huge\\bar{S}_{2,ABC}$', 
        marker=dict(color='black'), line =dict(width = 7.5),
        xaxis = 'x4', yaxis = 'y4', legend='legend4'
    )


    plt.update_layout(
        title =dict(text = 'Non Locality Indicators', font = dict(size = 40), x = 0), 
        xaxis = dict(title = '$\\huge t(\\omega_{0}^{-1})$', domain = (0,.425)),
        xaxis2 = dict(title = r'$\huge \frac{p_{A}}{E_{0}}$', domain = (0,.425) ),
        xaxis3 = dict(title = '$\\huge t(\\omega_{0}^{-1})$'), 
        xaxis4 = dict(title = '$\\huge t(\\omega_{0}^{-1})$'), 
        xaxis5 = dict(title = '$\\huge t(\\omega_{0}^{-1})$', domain = (.575, 1), anchor = 'y5'), 
        xaxis6 = dict(title = r'$\huge \frac{p_{A}}{E_{0}}$', domain = (.575, 1), anchor = 'y6'),

        yaxis = dict(title = '$\\huge \\mathcal{I}_{2,(AB)}$'),
        yaxis2=dict(title = r"$\huge \frac{p_{A}}{E_{0}}$"),
        yaxis3 = dict(title = r'$\huge S_{2,(AB)} $'),
        yaxis4 = dict(title = r'$\huge \mathcal{I}_{2,(AB:C)} $'),
        yaxis5 = dict(title = r'$\huge {S}_{2,(ABC)} $', domain = copy.copy(plt.layout['yaxis']['domain']), anchor = 'x5'),
        yaxis6 = dict(title = r'$\huge \frac{p_{A}}{E_{0}}$', domain = copy.copy(plt.layout['yaxis2']['domain']), anchor = 'x6'),

        height = 3500

    )
    titles = {
        '6':r'$\huge\text{{Two  Body  Entropy  Heat  Map}}(t(\omega_{{0}}^{{-1}})={t})$'.format(t = int(t[i])), 
        '4':r'$\huge\text{Three Body Entropy}$', 
        '3':r'$\huge\text{Three Body Mutual Information Estimate}$', 
        '5':r'$\huge\text{Two Body Entropy}$',
        '2':r'$\huge\text{{Two  Body Mutual  Information  Heat  Map}}(t(\omega_{{0}}^{{-1}})={t})$'.format(t = int(t[i])),
        '':r'$\huge\text{Two Body Mutual Information Estimate}$'.format(t[i]),
    }
    annotations = [
        dict(
            text=val,
            xref='paper',
            yref='paper',
            font = dict(size= 30),
            x = plt.layout[f'xaxis{key}']['domain'][0],
            y = plt.layout[f'yaxis{key}']['domain'][1]+.015,
            xanchor = 'left',
            yanchor = 'top',
        )
        for key, val in titles.items()
    ]
    annotations.extend((
        dict(
            text=r'$\huge\mathcal{I}_{2,(AB)}$',
            xref='paper',
            yref='paper',
            font = dict(size= 30),
            x = .475,
            y = plt.layout[f'yaxis2']['domain'][1]+.005,
            xanchor = 'left',
            yanchor = 'top',
        ),
        dict(
            text=r'$\huge S_{2,(AB)}$',
            xref='paper',
            yref='paper',
            font = dict(size= 30),
            x = 1.05,
            y = plt.layout[f'yaxis6']['domain'][1]+.005,
            xanchor = 'left',
            yanchor = 'top',
        )
    )
    )
    plt.update_layout(annotations = annotations )
    print('Non-Local Made')

    if(to_image):
        plt.to_image(os.path.join(plot_path, 'NonLocal.png'))
        return 
    else:
        return plt

def plot_ptrans(
        PhiBar:torch.Tensor, tfact:float, t:torch.Tensor, 
        plot_path:str, sun:SUConnection, B:LazyTensor,
        plt:PlotIt, u:LazyTensor, N:int, to_image:bool = True
    )->None:
    
    plt.reset()
    plt.subplot_grid(ncols=1, nrows=4, spacingr=.1, leg_opts=dict(bordercolor="black", borderwidth=3))
    
    plt.add_data(
        x = t, y = PhiBar.real.mean(1).pow(2).sum(-1).real, 
        marker = dict(color = 'blue'), line = dict(width = 4, dash = 'solid'),
        yaxis = 'y3', xaxis = 'x3',  legend = 'legend3',
        name = '$\\huge \\bar{\\Phi}_a\\bar{\\Phi}^a$', legendgroup = 'A', 
        legendgrouptitle_text = '$\\huge \\textbf{Order Parameter}$', showlegend = True, 
    )


    plt.add_data(
        x = t, y = u(tfact*t).real, 
        yaxis= 'y4', xaxis = 'x4', legend = 'legend4',
        line=dict(width = 4), marker = dict(color = 'purple'), 
        name = '$\\huge \\mu_{(t)}$', showlegend = True
    )
    plt.add_data(
        x = [0, t[-1]], y = [B(0)[:,2].real.mean().item(), B(0)[:,2].real.mean().item()],
        line=dict(width = 4, dash='dash'), marker=dict(color = 'orange'),   mode='lines',
        yaxis = 'y4', xaxis ='x4', legend = 'legend4', 
        name = '$\\huge|\\bar{B}|$', showlegend = True
    )
    S = oneBodyRenyiEntropy(PhiBar, sun = sun, )

    plt.extend_data(*(
            dict(
                x = t, y =  S[:,i],
                xaxis = 'x2', yaxis = 'y2', legend = 'legend2',
                line = dict(width = 10, dash = 'solid'), marker = dict(color = 'rgba(255,0,0,.25)',),
                showlegend = bool(i == 0), name = '$\\huge S_{2,A}$',legendgroup = 'S', legendgrouptitle_text = "$\\huge \\alpha = 2 \\textbf{Entropy}$"
            ) for i in range(N )
        )
    )
    plt.add_data(
        x = t, y=  S[:,].mean(-1),
        xaxis = 'x2', yaxis = 'y2', legend = 'legend2',
        line = dict(width = 7.5, dash='solid'), marker=dict(color='black'), 

        showlegend = True, name = '$\\huge \\bar{S}_{2}$', legendgroup = 'S'
    )
    J = torch.einsum('aij, qpij->qpa',sun.get_repr(), wigner_kernel(sun.n).conj() )/sun.n
    T = sun.get_repr()
    W = torch.einsum('...a, a, qpa->...qp', PhiBar.to(torch.complex128), sun.get_beta(T)[1:], J[...,1:]) + J[...,0]/sun.n 

    M2 = -(W.abs().pow(4).sum(dim=(-1,-2)).log()).real/3
    plt.extend_data(*(
            dict(
                x = t, y =  M2[:,i],
                xaxis = 'x', yaxis = 'y', legend = 'legend',
                line = dict(width = 10, dash = 'solid'), marker = dict(color = 'rgba(255,0,0,.25)',),
                showlegend = bool(i == 0), name = '$\\huge \\mathcal{M}_{4,A}$',legendgroup = 'M', 
                legendgrouptitle_text = "$\\huge \\alpha = 4 \\textbf{ Magic}$"
            ) for i in range(N )
        )
    )
    plt.add_data(
        x = t, y=  M2[:,].mean(-1),
        xaxis = 'x', yaxis = 'y', legend = 'legend',
        line = dict(width = 7.5, dash='solid'), marker=dict(color='black'),
        showlegend = True, name = '$\\huge \\bar{\\mathcal{M}}_{4}$', legendgroup = 'M'
    )
    plt.update_layout(
        height = 2500, width = 2000, 
        title = dict(
            text = r'$\huge \textbf{Dynamical Phase Transition  around  the  effective Daul Point}(\mu_{(t)} \sim |\bar{B}|)$',
            yref= 'paper', y = .975
        ),
        yaxis = dict(title = dict(text = r"$\huge \mathcal{M}_{4,A}$")),
        yaxis2= dict(title = dict(text = r"$\huge S_{2,A}$")),
        yaxis3= dict(title = dict(text = r"$\huge \bar{\Phi}_{a}\bar{\Phi}^{a}$")),
        yaxis4= dict(title = dict(text = r"$\huge Energy Scale$")),

        xaxis = dict(title = dict(text = r"$\huge t(\omega_{0}^{-1})$")),
        xaxis2= dict(title = dict(text = r"$\huge t(\omega_{0}^{-1})$")),
        xaxis3= dict(title = dict(text = r"$\huge t(\omega_{0}^{-1})$")),
        xaxis4= dict(title = dict(text = r"$\huge t(\omega_{0}^{-1})$")),
        
        legend = dict(title = dict(text = r"One-Body Magic"), font = dict(size = 25)),
        legend2 = dict(title = dict(text = r"One-Body Entropy"), font = dict(size = 25)),
        legend3 = dict(title = dict(text = r"Order Paramter"), font = dict(size = 25)),
        
    )

    if(to_image):
        plt.to_image(os.path.join(plot_path, 'PhaseTransition.png'))
        return 
    else:
        return plt


def plot_flav(PhiBar:torch.Tensor, sun:SUConnection, plt:PlotIt, t:torch.Tensor , path_base:str, N:int, to_image:bool = True)->None:
    Pz = pmnsrotPhi(PhiBar, sun)[...,]
    plt.reset()
    color = ['rgba(255,0,0,.5)', 'rgba(0,0,255,.5)']
    color_Bar = ['rgb(0,0,0)', 'rgb(100,50,150)']
    dash = ['dot','dash']
    for m, k in enumerate((torch.arange(2,sun.n+1).pow(2)-1)):
        plt.extend_data(*(
            dict(
                x = t, y = Pz[...,i,k].real, 
                marker=dict(color = color[m]), line = dict(width = 5), 
                showlegend = bool(i<1), 
                name = f'$\\huge \\Phi_{{A{str(int(k))}}}$'
            ) for i in range(N)
        ))
        plt.add_data(
            x = t, y = Pz[...,k].real.mean(-1), 
            marker=dict(color = color_Bar[m]), line = dict(width = 10, dash = dash[m]), 
            showlegend = True, name = f'$\\huge \\bar{{\\Phi}}_{str(int(k))}$'
        )
    plt.update_layout(
        xaxis = dict(title = dict(text = r'$\huge {t(\omega_{0}^{-1})}$')),
        yaxis = dict(title = dict(text = r"$\huge \text{Flavor Polarization}$")),
        title = dict(text = r"$\Huge \textbf{Flavor Polarization Evolution}$")
    )
    if(to_image):
        plt.to_image(os.path.join(path_base, 'Pz.png'))
        return 
    else:
        return plt