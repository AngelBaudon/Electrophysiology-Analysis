# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:59:29 2022

@author: Angel.BAUDON
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, glob
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D





def RadarMaker(names, ratios, savefig, show_fig = False):
    
    def Radar(number_of_variable, frame):
        theta = np.linspace(0, 2*np.pi, number_of_variable, endpoint=False)
    
        class RadarAxes(PolarAxes):
            name, RESOLUTION = 'radar', 1

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')
    
            def fill(self, *args, closed=True, **kwargs):
                return super().fill(closed=closed, *args, **kwargs)
    
            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                for line in lines: self._close_line(line)
    
            def _close_line(self, line):
                x, y = line.get_data()
                if x[0] != x[-1]: line.set_data(np.concatenate((x, [x[0]])),
                                                np.concatenate((y, [y[0]])))
    
            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)
    
            def _gen_axes_patch(self):
                return RegularPolygon((0.5, 0.5), number_of_variable, radius=.5, edgecolor="k")
    
            def _gen_axes_spines(self):
                spine = Spine(axes=self, spine_type='circle',
                              path=Path.unit_regular_polygon(number_of_variable))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
    
        register_projection(RadarAxes)
        return theta

    
    theta = Radar(len(ratios), 'polygon')
    fig, axes = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
    axes.set_rgrids(np.linspace(0, 1.2, 25))
    axes.plot(theta, ratios, c='g')
    axes.fill(theta, ratios, facecolor='limegreen', alpha=0.25)
    axes.set_varlabels(names), axes.set_ylim(0,1.2)
    plt.savefig(savefig)
    
    
folder = r"C:\Angel.BAUDON\Exp\ImCa\All exp\2023_03_CaIm 500 nM\Interdrug stat ratio"
for file in glob.glob(rf'{folder}\*.xlsx'):
    df = pd.read_excel(file).drop('Treatment', axis=1)
    features = list(df.columns)
    
    pval = df.values[0]
    pval = [1-x for x in pval]
    print(pval)

    RadarMaker(features, pval, rf'{file[:-5]}.pdf')

