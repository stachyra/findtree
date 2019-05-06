#!/usr/bin/env python

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from findtree import findtree
from math import floor

# Draws the saturation / value triangles used in one of the figures
def make_sv_triangle(ax):
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]    
    half = 0.005
    # Inside of each saturation-value triangle, draw polygons with their
    # face colors selected to represent the actual value and saturation
    # corresponding to their location
    for ival in np.arange(half, 1+half, 2*half):
          for isat in np.arange(half, ival+half, 2*half):
              if isat < ival:
                  finalsat = isat+half
              else:
                  finalsat = isat-half
              vertices = [[isat-half, ival-half],
                          [isat-half, ival+half],
                          [isat+half, ival+half],
                          [finalsat, ival-half],
                          [0,0]]
              fc = colors.hsv_to_rgb([0.167, isat, ival])
              pp = PathPatch(Path(vertices, codes), facecolor=fc,
                             edgecolor='none', zorder=-1)
              ax.add_patch(pp)
    # Plot triangle hypoteneuse and adjust axes and tick labels
    ax.plot([0.0, 1.0], [0.0, 1.0], color=[0,0,0], zorder=2)
    ax.set_xlim(0, 1)
    ax.xaxis.tick_top()
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')

# Image files to process
fname = ['nmzwj.png', 'aVZhC.png', '2K9EF.png',
         'YowlH.png', '2y4o5.png', 'FWhSP.png']

# Corners of figure zoom locations
zoomloc = [[45, 157], [170, 265], [320, 795],
           [390, 820], [100, 615], [80, 1291]]
# Zoom box size in pixels
zoomsize = 100

# Initialize figures
fig, ax = {}, {}
gs = []
fginfo = {'raw':    {'title': 'Input Photos',
                     'size': (16,7)},
          'mnhst':  {'title': 'Grayscale Brightness Histogram',
                     'size': (16,7)},
          'satval': {'title': 'Saturation-Value Scatterplot',
                     'size': (12,7)},
          'huehst': {'title': 'Saturated Hues Histogram',
                     'size': (16,7)},
          'allplt': {'title': 'All Color and Brightneess Plots',
                     'size': (16,9)},
          'thresh': {'title': 'Thresholded HSV and Monochrome Brightness',
                     'size': (16,7)},
          'clust':  {'title': 'DBSCAN Clusters (Raw Pixel Output)',
                     'size': (16,7)},
          'cltwo':  {'title': 'DBSCAN Clusters (Slightly Dilated for Display)',
                     'size': (16,7)},
          'border': {'title': 'Trees with Borders',
                     'size': (16,7)},
          'svexpl': {'title': 'Saturation Value Example',
                     'size': (4,4)}}
for k in fginfo:
    fig[k] = plt.figure(figsize=fginfo[k]['size'], facecolor='w')
    fig[k].canvas.set_window_title(fginfo[k]['title'])
    ax[k] = []

# Color ordering for displaying cluster results
plcol = np.array([[1.0, 0.0, 0.0],  # red
                  [0.0, 0.7, 0.0],  # green
                  [1.0, 0.5, 0.0],  # orange
                  [0.9, 0.0, 0.9],  # magenta
                  [0.0, 0.0, 1.0],  # blue
                  [0.8, 0.8, 0.8],  # gray
                  [0.0, 0.0, 0.0]]) # black
ncol = plcol.shape[0]

axshare = {}
# Substitute slightly non-default hue thresholds for the findtree algorithm
hlt, hrt = 0.2, 0.85
# Saturation, value, and brightness thresholds which are already coded as
# algorithm defaults
sthresh, vthresh, bthresh = 0.7, 0.7, 220
for ii, name in zip(range(len(fname)), fname):
    # Open the file and convert to rgb image
    rgbimg = np.asarray(Image.open(name))
    ax['raw'].append(fig['raw'].add_subplot(2,3,ii+1))
    ax['raw'][ii].set_axis_off()
    ax['raw'][ii].imshow(rgbimg, interpolation='nearest')
    ax['raw'][ii].set_xlim(0, rgbimg.shape[1]-1)
    ax['raw'][ii].set_ylim(rgbimg.shape[0], -1)
    imgshape = rgbimg.shape

    # Prepare the grid layout for the master plot ('allplt') which will
    # show a cropped version of each initial photo, plus the grayscale
    # histograms, saturation-value scatterplots, and hue histograms, all
    # together in a single figure
    hdm, vdm = fginfo['allplt']['size']
    hborder = 0.025
    vborder = hborder * hdm / vdm
    hmajspace = 0.035
    vmajspace = hmajspace * hdm / vdm
    hgrddim = (1.0 - 2 * hborder - 2 * hmajspace) / 3
    vgrddim = (1.0 - 2 * vborder - vmajspace) / 2
    left = hborder + (ii % 3) * (hgrddim + hmajspace)
    right = hborder + (1.0 + (ii % 3)) * hgrddim + (ii % 3) * hmajspace
    top = 1.0 - vborder - floor(ii/3) * (vgrddim + vmajspace)
    bottom = 1.0 - vborder - (1.0 + floor(ii/3)) * vgrddim - floor(ii/3) * vmajspace
    hcrct = hgrddim
    vcrct = vgrddim * vdm / hdm
    hratio = [vcrct/2, hcrct-vcrct/2]
    gs.append(fig['allplt'].add_gridspec(2, 2, left=left, right=right,
                                         top=top, bottom=bottom,
                                         width_ratios=hratio))
    # Create the upper left set of axes among each panel of four, and add
    # the cropped version of each photo to the newly created axes
    ax['allplt'].append(fig['allplt'].add_subplot(gs[-1][0,0]))
    ax['allplt'][-1].set_axis_off()
    x, y = zoomloc[ii]
    ax['allplt'][-1].imshow(rgbimg[x:x+zoomsize,y:y+zoomsize],
                              interpolation='nearest')

    # Show grayscale histograms
    grayvals = np.asarray(Image.fromarray(rgbimg).convert('L')).flatten()
    edges = np.linspace(-0.5, 255.5, 257)
    if ii == 0:
        ax['mnhst'].append(fig['mnhst'].add_subplot(2,3,ii+1))
        axshare['mnhst'] = ax['mnhst'][0]
        ax['allplt'].append(fig['allplt'].add_subplot(gs[-1][0,1]))
        axshare['allplt'] = {'mnhst': ax['allplt'][-1]}
    else:
        ax['mnhst'].append(fig['mnhst'].add_subplot(2,3,ii+1,
                                                    sharex=axshare['mnhst'],
                                                    sharey=axshare['mnhst']))
        ax['allplt'].append(fig['allplt'].add_subplot(gs[-1][0,1],
                                                      sharex=axshare['allplt']['mnhst'],
                                                      sharey=axshare['allplt']['mnhst'])) 
    for ftype in ('mnhst', 'allplt'):          
        N, junk, patches = ax[ftype][-1].hist(grayvals, bins=edges)
        normlim = colors.Normalize(0, 255)
        # Shade the area inside the histogram from black to gray to white
        for thisval, thispatch in zip(edges[0:256]+0.5, patches):
            thispatch.set_facecolor(cm.gray(normlim(thisval)))
        junk1, junk2, ptch = ax[ftype][-1].hist(grayvals, bins=edges,
                                                  histtype='step')
        ptch[0].set_edgecolor([0.0, 0.0, 0.0])
        # Draw a red line at the default brightness threshold
        ax[ftype][-1].axvline(bthresh, color=[1,0,0])
        ax[ftype][-1].set_xlim(-0.5, 255.5)
        ax[ftype][-1].set_yscale('log')
    # Print percentage of pixels in each image falling above and below the
    # brightness threshold
    pct = len(np.where(grayvals > bthresh)[0]) / len(grayvals) * 100
    ax['mnhst'][-1].text(224, 1.8e5, "{:.2f}%".format(pct),
                         fontsize='medium', fontweight='bold',
                         horizontalalignment='left')
    ax['mnhst'][-1].text(216, 1.8e5, "{:.2f}%".format(100-pct),
                         fontsize='medium', fontweight='bold',
                         horizontalalignment='right')


    # Show saturation-value scatterplot
    ax['satval'].append(fig['satval'].add_subplot(2,3,ii+1))
    ax['allplt'].append(fig['allplt'].add_subplot(gs[-1][1,0]))
    for ftype in ('satval', 'allplt'):
        # Make the basic background triangle for each saturation-value plot
        make_sv_triangle(ax[ftype][-1])
        # Reshape the RGB image as a long N X 3 list of colors
        newshape = [imgshape[0]*imgshape[1], imgshape[2]]
        hsvpix = colors.rgb_to_hsv(rgbimg.astype(float)/255).reshape(newshape)
        # In a 2D scatterplot, plot the value and saturation of each pixel
        # in the image
        ax[ftype][-1].scatter(np.multiply(hsvpix[:,1], hsvpix[:,2]),
                                             hsvpix[:,2],
                                             c=np.asarray([[0, 0.25, 1]]),
                                             marker='.', s=0.1, zorder=0)
        # Plot thresholds in red
        ax[ftype][-1].plot([0.0, vthresh], [sthresh, vthresh], color=[1,0,0], zorder=1)
        ax[ftype][-1].plot([0.0, vthresh], [0.0, 1.0], color=[1,0,0], zorder=1)
    # Print percentage of pixels that ended up above or below each threshold
    pct = {}
    pct['lo'] = {'lo': len(np.where(np.logical_and(hsvpix[:,1] <= sthresh,
                                                   hsvpix[:,2] <= vthresh))[0]) / hsvpix.size * 100}
    pct['lo']['hi'] = len(np.where(np.logical_and(hsvpix[:,1] <= sthresh,
                                                  hsvpix[:,2] > vthresh))[0]) / hsvpix.size * 100
    pct['hi'] = {'lo': len(np.where(np.logical_and(hsvpix[:,1] > sthresh,
                                                   hsvpix[:,2] <= vthresh))[0]) / hsvpix.size * 100}
    pct['hi']['hi'] = len(np.where(np.logical_and(hsvpix[:,1] > sthresh,
                                                  hsvpix[:,2] > vthresh))[0]) / hsvpix.size * 100
    xpos = {'lo': {'lo': 0.18, 'hi': 0.47}, 'hi': {'lo': 0.4, 'hi': 0.7}}
    ypos = {'lo': {'lo': vthresh-0.03, 'hi': vthresh-0.03}, 'hi': {'lo': 0.97, 'hi': 0.97}}
    for isat in ('lo', 'hi'):
        for ival in ('lo', 'hi'):
            ax['satval'][-1].text(xpos[ival][isat], ypos[ival][isat],
                                  "{:.2f}%".format(pct[ival][isat]),
                                  fontsize='medium',
                                  fontweight='bold',
                                  horizontalalignment='left',
                                  verticalalignment='top')

    # Show saturated hue histogram
    if ii == 0:
        ax['huehst'].append(fig['huehst'].add_subplot(2,3,ii+1))
        axshare['huehst'] = ax['huehst'][0]
        ax['allplt'].append(fig['allplt'].add_subplot(gs[-1][1,1]))
        axshare['allplt']['huehst'] = ax['allplt'][-1]
    else:
        ax['huehst'].append(fig['huehst'].add_subplot(2,3,ii+1,
                                                      sharex=axshare['huehst'],
                                                      sharey=axshare['huehst']))
        ax['allplt'].append(fig['allplt'].add_subplot(gs[-1][1,1],
                                                      sharex=axshare['allplt']['huehst'],
                                                      sharey=axshare['allplt']['huehst'])) 
    svidx = np.logical_and((hsvpix[:,1] > sthresh), (hsvpix[:,2] > vthresh))
    satcol = hsvpix[np.where(svidx),0][0]
    edges = np.linspace(0.0, 1.0, 101)
    for ftype in ('huehst', 'allplt'):
        N, junk, patches = ax[ftype][-1].hist(satcol, bins=edges)
        # Color inside the histogram to show the actual hues represented 
        # by each hue value
        for thisval, thispatch in zip(edges[0:100]+0.005, patches):
            thispatch.set_facecolor(cm.hsv(thisval))
        junk1, junk2, ptch = ax[ftype][-1].hist(satcol, bins=edges,
                                                histtype='step')
        ptch[0].set_edgecolor([0.0, 0.0, 0.0])
        # Draw red lines corresponding to left and right thresholds
        ax[ftype][-1].axvline(hlt, color=[1,0,0])
        ax[ftype][-1].axvline(hrt, color=[1,0,0])
        ax[ftype][-1].set_xlim(0.0, 1.0)
        ax[ftype][-1].set_yscale('log')
    pct = {}
    svhiidx = np.logical_and(hsvpix[:,1] > sthresh, hsvpix[:,2] > vthresh)
    # Calculate the percentages of pixels falling into each region of the
    # histogram, relative to the left and right hue threasholds, so that we
    # can display the info in an annotation.
    pct['lo'] = len(np.where(np.logical_and(svhiidx,
                                            hsvpix[:,0] < hlt))[0]) / hsvpix.size * 100
    hmdidx = np.logical_and(hsvpix[:,0] >= hlt, hsvpix[:,0] <= hrt)
    pct['md'] = len(np.where(np.logical_and(svhiidx, hmdidx))[0]) / hsvpix.size * 100
    pct['hi'] = len(np.where(np.logical_and(svhiidx,
                                            hsvpix[:,0] > hrt))[0]) / hsvpix.size * 100
    xpos = {'lo': 0.03, 'md': 0.45, 'hi': 0.86}
    for ihue in ('lo', 'md', 'hi'):
        ax['huehst'][-1].text(xpos[ihue], 2.5e5, "{:.3f}%".format(pct[ihue]),
                              fontsize='medium', fontweight='bold',
                              horizontalalignment='left')

    # Get the tree borders as well as a bunch of other intermediate values
    # that will be used to illustrate how the algorithm works
    borderseg, X, labels, Xslice = findtree(rgbimg, hueleftthr=hlt,
                                            huerightthr=hrt)
    labels, Xslice = np.asarray(labels), np.asarray(Xslice)

    # Display thresholded images
    ax['thresh'].append(fig['thresh'].add_subplot(2,3,ii+1))
    ax['thresh'][ii].set_xticks([])
    ax['thresh'][ii].set_yticks([])
    binimg = np.zeros((rgbimg.shape[0], rgbimg.shape[1]))
    for v, h in X:
        binimg[v,h] = 255
    ax['thresh'][ii].imshow(binimg, interpolation='nearest', cmap='Greys')

    # Display color-coded clusters
    # 'clust' == Raw version
    ax['clust'].append(fig['clust'].add_subplot(2,3,ii+1))
    ax['clust'][ii].set_xticks([])
    ax['clust'][ii].set_yticks([])
    # 'cltwo' == Dilated slightly for display only
    ax['cltwo'].append(fig['cltwo'].add_subplot(2,3,ii+1))
    ax['cltwo'][ii].set_xticks([])
    ax['cltwo'][ii].set_yticks([])
    ax['cltwo'][ii].imshow(binimg, interpolation='nearest', cmap='Greys')
    clustimg = np.ones(rgbimg.shape)
    # Labels tell which cluster each pixel belongs to, and the count value
    # tells the relative size of each cluster, in terms of the number of
    # pixels that belong to it.  A label of -1 means that a pixel is not
    # assigned to any cluster.   
    unique_labels, count = np.unique(labels, return_counts=True)
    # For pixels that weren't assigned to any cluster, make their count
    # value negative, so that when we sort based upon count, the unassigned
    # pixels get pushed to the end of the list, regardless of how many there
    # were.
    count[np.where(unique_labels < 0)] *= -1
    sort_labels = unique_labels[np.flip(np.argsort(count))]
    for lblidx, srtlbl in zip(range(len(sort_labels)), sort_labels):
        # Pixels that weren't assigned to any cluster get plotted using the
        # final color in the colormap (black)
        if srtlbl == -1:
            colidx = ncol-1
        # Cluster labels that exceed the number of available colors get
        # plotted using the second-to-last color in the colormap (gray)
        elif lblidx > ncol-3:
            colidx = ncol-2
        # Otherwise, all other clusters are assigned a position in the
        # colormap in a specific sequence in order of size
        else:
            colidx = lblidx
        pixidx = np.where(labels==srtlbl)[0]
        # Plot each cluster using a color chosen based on size ordering,
        # from largest to smallest 
        clustimg[Xslice[pixidx,0],Xslice[pixidx,1],:] = plcol[colidx,:]
        # Use dilated (artificially enlarged) pixel sizes, for display
        # purposes only
        ax['cltwo'][ii].plot(Xslice[pixidx,1], Xslice[pixidx,0], 'o',
                             markerfacecolor=plcol[colidx,:], markersize=2,
                             markeredgecolor=plcol[colidx,:])
    ax['clust'][ii].imshow(clustimg)
    ax['cltwo'][ii].set_xlim(0, binimg.shape[1]-1)
    ax['cltwo'][ii].set_ylim(binimg.shape[0], -1)

    # Plot original images with red borders around the trees
    ax['border'].append(fig['border'].add_subplot(2,3,ii+1))
    ax['border'][ii].set_axis_off()
    ax['border'][ii].imshow(rgbimg, interpolation='nearest')
    for vseg, hseg in borderseg:
        ax['border'][ii].plot(hseg, vseg, 'r-', lw=3)
    ax['border'][ii].set_xlim(0, binimg.shape[1]-1)
    ax['border'][ii].set_ylim(binimg.shape[0], -1)

# Adjust subplot spacing and margins
LFT, RGT, BOT, TOP, WSP, HSP = 0.015, 0.985, 0.015, 0.985, 0.05, 0.05
for k in ['raw', 'thresh', 'clust', 'cltwo', 'border']:
    fig[k].subplots_adjust(left=LFT, right=RGT, bottom=BOT, top=TOP,
                           wspace=WSP, hspace=HSP)
fig['mnhst'].tight_layout()
fig['satval'].tight_layout()
fig['huehst'].tight_layout()

# Plot an example saturation value triangle
ax['svexpl'].append(fig['svexpl'].add_subplot(1,1,1))
make_sv_triangle(ax['svexpl'][-1])
ax['svexpl'][-1].set_xticks([])
ax['svexpl'][-1].set_yticks([])

plt.show()
