#Purpose: 
#Category: 
#Authors: Dale Weigt (D.M.Weigt@soton.ac.uk), apadpted from Randy Gladstone's 'gochandra' IDL script


"""All the relevant packages are imported for code below"""

import numpy as np
import scipy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
import datetime
from datetime import datetime, timedelta




"""Setup the font used for plotting"""

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['xtick.labelsize']=14
matplotlib.rcParams['ytick.labelsize']=14
matplotlib.rcParams['agg.path.chunksize'] = 1000000



"""Below are a list of defined functions used in the code below
----------------------------------------------------------------
(Look into providing help pages plus errors - look at Hull stuff)"""

def format_e(n):
    
    """Takes a selected number (float/int) and converts it into scientifc notation.
    
    The number is printed as a string and can be used for calculations, headings, axis labels etc..."""
    
    a = '%E' % n # converts n into scientifc notaion
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1] # formats number to 3 sig figs with scientifc notation

def doy_frac(DOY, hour, minutes, seconds): 
    
    """The function takes the integer Day of Year (DOY), hour, minutes and seconds of the observation (inputted by the user) and calculates
    the DOY with fractional part for the observation time"""
    
    frac = DOY + hour/24.0 + minutes/1440.0 + seconds/86400.0 #calculation for the correct DOY fraction.
    return frac

def findcosmu(re0, rp0, sublat, latc, lon): # considers latc to be plaentocentric latitudes, but sublat to be planetographic 
    
    """Takes the equitorial and polar radius of Jupiter (re0, rp0 respectively), the sub-latitude of Jupiter, latitude and
    longitude (both in radians) to determine the "cos(mu)" of the photons. This effectively helps to idenify where the limb 
    of Jupiter occurs in the Chandra observations"""
    
    rfactor = (re0/rp0)**2 # ratio  of the equitorial radius and polar radius...
    lat = np.arctan(np.tan(latc)*rfactor) # and coordinate transformation from planetocentric latitude -> planetographic latitude
    ans = (rfactor * (np.cos(lon)*np.cos(sublat)*np.cos(lat)) + (np.sin(sublat)*np.sin(lat))) / np.sqrt(rfactor*np.cos(sublat)**2 \
            + np.sin(lat)**2) / np.sqrt(rfactor * np.cos(lat)**2 + np.sin(lat)**2) # to return the value(s) of cos(mu)
    return ans

def ltln2xy(alt, re0, rp0, r, e, h, phi1, phig, lambda0, p, d, gamma, omega, latc, lon): # Considers latc to be planetocentric latitudes, but phi1
    # and phi1 to be phig to be planetographic
    
    """Takes the latitude and longtitude of the seleced events and performs an S3 coordinate transformation. The (x,y) positions, 
    cosc and the element position and length of the array that have cosc >= 1/(ratio of dist to Jupiter and euqitorial radius [in pixels]
    of the photons are returned after the transformation has been carried out"""
    
    rfactor = (re0/rp0)**2
    lat = np.arctan(np.tan(latc)*rfactor)
    n = r / np.sqrt(1.0 - (e*np.sin(lat))**2)
    c = ((n+alt) / r)*np.cos(lat)
    s = ((n*(1.0 - e**2) + alt) / r) * np.sin(lat)
    k = h / (p*np.cos(phi1-phig) - s*np.sin(phi1) - c*np.cos(phi1) * np.cos(lon-lambda0))
    cosc = s*np.sin(phi1) + c*np.cos(phi1) * np.cos(lon-lambda0)
    x = -k*c*np.sin(lon-lambda0)
    y = k*(p*np.sin(phi1-phig) + s * np.cos(phi1) - c*np.sin(phi1) * np.cos(lon-lambda0))
    a = ((y*np.cos(gamma) + x*np.sin(gamma)) * np.sin(omega) / h) + np.cos(omega)
    xt = (x*np.cos(gamma) - y*np.sin(gamma)) * np.cos(omega) / a
    yt = (y*np.cos(gamma) + x*np.sin(gamma)) / a - omega * (d-r)
    condition = np.where(cosc >= 1.0/p)[0]
    count = np.count_nonzero(condition)
    
    return xt, yt, cosc, condition, count


def selecttime(time):
    
    """Returns the beginning and end time of the observation as well as the offset"""
    
    num = len(time) # length of the time array
    begt, endt = [time[0], time[num-1]] # defines the beginning and end time of the observation
    offset = time-time[0] # finds the offset of the observation and defines the times of the observation from this (i.e start from
    # 0 - begining of observation - and finished on the final elpased time of the Chandra observations).
  
    return begt, endt, offset


def select_region(x1,x2, y1,y2, bigxarr, bigyarr, bigchannel, cha_min, cha_max):
    
    """The user inputs the area of the region they want to analyse of the form [x1,x2], [y1,y2] (where x1 < x2, y1 < y2)
    
    ***IF THE USER WANTS TO CHANGE THE TEXT FILE NAME, HARDWIRE THE NAME WITHIN THE np.savetxt("") FUNCTION***"""

    selected_region_x, selected_region_y = [x1,x2], [y1,y2] #...the positions of the rectangle is then saved as an array.
    indx = np.where((bigchannel >= cha_min) & (bigchannel <= cha_max) & (bigxarr >= selected_region_x[0]) &\
                    (bigxarr <= selected_region_x[1]) & (bigyarr >= selected_region_y[0]) & (bigyarr <= selected_region_y[1]))[0]
    
    # finds which photons lie within the selected region between a given pha channel range...  
    
     
    #np.savetxt("%s_selected_region_test.txt" % obs_id \
    #           , np.c_[bigxarr[indx], bigyarr[indx], bigtime[indx], bigchannel[indx]])
    #...and records it in a text file with the name automatically generated unless altered by user
    
    return indx
    
    
def plotprops(props,st,units,sup_lon_list,sup_lat_list,obs_id, custom_map):
          
    """Plots the X-ray emission in given units across Jupiter's surface on a S3 longitude vs. latitude grid. The Io and magnetotail footprints are also plotted. The inputs are defined as follows:
    
    props:   the property of the X-rays plotted onto the map. For example, Chandra code provides a flux, exposure time and a brightness (using a conversion factor).
    
    st:     string for the title of the plot.
    
    units:  units of the color bar used in the map.
    
    sup_lon_list:   the list of longitudes for the X-ray emission found.
    
    sup_lat_list:   the list of latitudes for the X-ray emission found.
    
    obs_id:  the observation id - this is set as default
    
    custom_map:     custom color map created from custom c_map code"""
    
    geomsy_data = scipy.io.readsav('geomsy.sav')
    lam_vip = geomsy_data['lam_vip'] 
    rthet_vip = geomsy_data['rthet_vip'] 
    rns_vip = geomsy_data['rns_vip'] 
    lam306_vip = geomsy_data['lam306_vip'] 
    r306thet_vip = geomsy_data['r306thet_vip'] 
    r306_vip = geomsy_data['r306_vip']
    
    fig, axes=plt.subplots(figsize=(12,8))
    axes = plt.gca()
    axes.set_xlim(360,0)
    axes.set_ylim(-90,90)
    axes.plot([0,359], [-90,90],linestyle='None')
    axes.set_title(st)
    axes.text(175, -115, 'Max = %f %s' % (np.amax(props), units), fontsize=15, horizontalalignment='center')

    lngarr = np.arange(360)
    
    max_cbar = np.amax(props)
    
    mesh=axes.pcolormesh(np.arange(0,360), np.arange(-90,91), props.T, norm=colors.PowerNorm(gamma=0.75), vmin=0, vmax = max_cbar,\
                        alpha = 0.75, cmap=custom_map)
    
    cbar = plt.colorbar(mesh)
    cbar.set_label('%s' % units, rotation=270, labelpad = 20)
    
    axes.scatter(sup_lon_list, sup_lat_list - 90, color='white', edgecolors = 'black', label='Chandra',alpha=0.5)
    #axes.legend(loc='center left', bbox_to_anchor=(1.22, 0.5))
    
    for i in range(0,361,20):
        axes.plot([i,i], [-90,90], linestyle = 'dashed', color = 'white', alpha=0.4)
        
    for i in range(0,181,20):
        axes.plot([0,360], [i-90, i-90], linestyle='dashed', color='white', alpha=0.4)

 
    
    for i in range(0,2):
        lng = lam306_vip[i,:]
        lat = r306thet_vip[i,:]
#        print(lng)
        dl = lng[1:36] - lng[0:35]
#        print(dl)
        p_cond = np.where(abs(dl) > 180)[0]
#        print(p_cond)
        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white')
        
        lng = lam_vip[i,:]
        lat = rthet_vip[i,:]
        dl - lng[1:36] - lng[0:35]
        p_cond = np.where(abs(dl) > 180)[0]
        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white', linestyle='dashed')
        
        axes.set_xlabel('S3 Longitude (Degrees)')
        axes.set_ylabel('Latitude (Degrees)')
    
    folder = input('Enter file path of folder to save plot (map): ')
    #plt.savefig('%s_%s_map.png' % (obs_id,st))
    plt.savefig('%s/%s_map.png' %(folder,obs_id))
    plt.close()
    return 

def plotprops_region_select(props,st,units,sup_lon_list,sup_lat_list,obs_id, custom_map):
    
    """Plots the X-ray emission in given units across Jupiter's surface on a S3 longitude vs. latitude grid. The code prompts the user to select the range of S3 longitudes and latitudes to map. The Io and magnetotail footprints are also plotted. The inputs are defined as follows:
    
    props:   the property of the X-rays plotted onto the map. For example, Chandra code provides a flux, exposure time and a brightness (using a conversion factor).
    
    st:     string for the title of the plot.
    
    units:  units of the color bar used in the map.
    
    sup_lon_list:   the list of longitudes for the X-ray emission found.
    
    sup_lat_list:   the list of latitudes for the X-ray emission found.
    
    obs_id:  the observation id - this is set as default
    
    custom_map:     custom color map produced from the c_map script."""
        
    geomsy_data = scipy.io.readsav('geomsy.sav')
    lam_vip = geomsy_data['lam_vip'] 
    rthet_vip = geomsy_data['rthet_vip'] 
    rns_vip = geomsy_data['rns_vip'] 
    lam306_vip = geomsy_data['lam306_vip'] 
    r306thet_vip = geomsy_data['r306thet_vip'] 
    r306_vip = geomsy_data['r306_vip']
    x_lim_1 = int(input("Enter lower S3 lon limit: "))
    x_lim_2 = int(input("Enter upper S3 lon limit: "))
    y_lim_1 = int(input("Enter lower lat limit: "))
    y_lim_2 = int(input("Enter upper lat limit: "))
    
    fig, axes=plt.subplots(figsize=(12,8))
    axes = plt.gca()
    axes.set_xlim(x_lim_2,x_lim_1)
    axes.set_ylim(y_lim_1,y_lim_2)
    axes.plot([0,359], [-90,90],linestyle='None')
    axes.set_title(st)
    axes.text((x_lim_1 + x_lim_2)/2, y_lim_1-5, 'Max = %f %s' % (np.amax(props), units), fontsize=15, horizontalalignment='center')

    lngarr = np.arange(360)
    
    max_cbar = np.amax(props)
    mesh=axes.pcolormesh(np.arange(0,360), np.arange(-90,91), props.T, norm=colors.PowerNorm(gamma=0.75), vmin=0, vmax = max_cbar,\
                        alpha = 0.75, cmap=custom_map)
    
    cbar = plt.colorbar(mesh)
    cbar.set_label('%s' % units, rotation=270, labelpad = 20)
    
    axes.scatter(sup_lon_list, sup_lat_list - 90, color='white', edgecolors = 'black', label='Chandra',alpha=0.5)
    #axes.legend(loc='center left', bbox_to_anchor=(1.22, 0.5))
 
    for i in range(0,361,20):
        axes.plot([i,i], [-90,90], linestyle = 'dashed', color = 'white', alpha=0.4)
    
    for i in range(0,361,5):
        axes.plot([i,i], [-90,90], linestyle = 'dashed', color = 'cyan', alpha=0.8)
        
    for i in range(0,181,20):
        axes.plot([0,360], [i-90, i-90], linestyle='dashed', color='white', alpha=0.4)
        
    for i in range(0,181,5):
        axes.plot([0,360], [i-90, i-90], linestyle='dashed', color='cyan', alpha=0.8)    

 
    
    for i in range(0,2):
        lng = lam306_vip[i,:]
        lat = r306thet_vip[i,:]

        dl = lng[1:36] - lng[0:35]

        p_cond = np.where(abs(dl) > 180)[0]

        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white')
        
        lng = lam_vip[i,:]
        lat = rthet_vip[i,:]
        dl - lng[1:36] - lng[0:35]
        p_cond = np.where(abs(dl) > 180)[0]
        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white', linestyle='dashed')
        
        axes.set_xlabel('S3 Longitude (Degrees)')
        axes.set_ylabel('Latitude (Degrees)')
    
    folder = input('Enter file path of folder to save plot (map): ')
    #plt.savefig('%s_%s_map.png' % (obs_id,st))
    plt.savefig('%s/%s_map_region_select.png' %(folder,obs_id))
    #plt.close()
    return x_lim_1, x_lim_2, y_lim_1, y_lim_2

def plotprops_time(props,st,units,sup_lon_list,sup_lat_list,obs_id):
    geomsy_data = scipy.io.readsav('geomsy.sav')
    lam_vip = geomsy_data['lam_vip'] 
    rthet_vip = geomsy_data['rthet_vip'] 
    rns_vip = geomsy_data['rns_vip'] 
    lam306_vip = geomsy_data['lam306_vip'] 
    r306thet_vip = geomsy_data['r306thet_vip'] 
    r306_vip = geomsy_data['r306_vip']
    
    fig, axes=plt.subplots(figsize=(12,8))
    axes = plt.gca()
    axes.set_xlim(360,0)
    axes.set_ylim(-90,90)
    axes.plot([0,359], [-90,90],linestyle='None')
    axes.set_title(st)
    axes.text(175, -115, 'Max = %f %s' % (np.amax(props), units), fontsize=15, horizontalalignment='center')

    lngarr = np.arange(360)
    
    max_cbar = np.amax(props)
    mesh=axes.pcolormesh(np.arange(0,360), np.arange(-90,91), props.T, norm=colors.PowerNorm(gamma=0.75), vmin=0, vmax = max_cbar,\
                        alpha = 0.75, cmap='Blues')
    
    cbar = plt.colorbar(mesh)
    cbar.set_label('%s' % units, rotation=270, labelpad = 20)
    
    axes.scatter(sup_lon_list, sup_lat_list - 90, color='white', edgecolors = 'black', label='Chandra')
    #axes.legend(loc='center left', bbox_to_anchor=(1.22, 0.5))
    
    for i in range(0,361,20):
        axes.plot([i,i], [-90,90], linestyle = 'dashed', color = 'white', alpha=0.4)
        
    for i in range(0,181,20):
        axes.plot([0,360], [i-90, i-90], linestyle='dashed', color='white', alpha=0.4)

 
    
    for i in range(0,2):
        lng = lam306_vip[i,:]
        lat = r306thet_vip[i,:]

        dl = lng[1:36] - lng[0:35]

        p_cond = np.where(abs(dl) > 180)[0]

        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white')
        
        lng = lam_vip[i,:]
        lat = rthet_vip[i,:]
        dl - lng[1:36] - lng[0:35]
        p_cond = np.where(abs(dl) > 180)[0]
        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white', linestyle='dashed')
        
        axes.set_xlabel('S3 Longitude (Degrees)')
        axes.set_ylabel('Latitude (Degrees)')
    
    folder = input('Enter file path of folder to save plot (timemap): ')
    #plt.savefig('%s_%s_map.png' % (obs_id,st))
    plt.savefig('%s/%s_timemap.png' %(folder,obs_id))
    plt.close()
    return

def plotprops_int(iteration,props,st,units,sup_lon_list,sup_lat_list,obs_id, custom_map, folder):
    
    """Plots the X-ray emission in given units across Jupiter's surface on a S3 longitude vs. latitude grid. Multiple maps are produced depending on the number of intervals the obsrvation has baan split into. The Io and magnetotail footprints are also plotted. The inputs are defined as follows:
    
    props:   the property of the X-rays plotted onto the map. For example, Chandra code provides a flux, exposure time and a brightness (using a conversion factor).
    
    iteration:      the counter used to map all time interavls of the Chandra observation.
    
    st:     string for the title of the plot.
    
    units:  units of the color bar used in the map.
    
    sup_lon_list:   the list of longitudes for the X-ray emission found.
    
    sup_lat_list:   the list of latitudes for the X-ray emission found.
    
    obs_id:  the observation id - this is set as default
    
    custom_map:     custom color map produced from the c_map script.
    
    folder:   the file path for the saved plots
    """
        
    geomsy_data = scipy.io.readsav('geomsy.sav')
    lam_vip = geomsy_data['lam_vip'] 
    rthet_vip = geomsy_data['rthet_vip'] 
    rns_vip = geomsy_data['rns_vip'] 
    lam306_vip = geomsy_data['lam306_vip'] 
    r306thet_vip = geomsy_data['r306thet_vip'] 
    r306_vip = geomsy_data['r306_vip']
    
    fig, axes=plt.subplots(figsize=(12,8))
    axes = plt.gca()
    axes.set_xlim(360,0)
    axes.set_ylim(-90,90)
    axes.plot([0,359], [-90,90],linestyle='None')
    axes.set_title(st)
    axes.text(175, -115, 'Max = %f %s' % (props.max(), units), fontsize=15, horizontalalignment='center')

    lngarr = np.arange(360)
    
    max_cbar = np.amax(props)
    mesh=axes.pcolormesh(np.arange(0,360), np.arange(-90,91), props.T, norm=colors.PowerNorm(gamma=0.75), cmap=custom_map, vmin=0,\
                         vmax = props.max(), alpha = 0.75)
    
    cbar = plt.colorbar(mesh)
    cbar.set_label('%s' % units, rotation=270, labelpad = 20)
    
    axes.scatter(sup_lon_list[iteration], sup_lat_list[iteration] - 90, color='black', edgecolors = 'black', label='Chandra', alpha=0.35)
    #axes.legend(loc='center left', bbox_to_anchor=(1.22, 0.5))
    
    for i in range(0,361,20):
        axes.plot([i,i], [-90,90], linestyle = 'dashed', color = 'white', alpha=0.4)
        
    for i in range(0,181,20):
        axes.plot([0,360], [i-90, i-90], linestyle='dashed', color='white', alpha=0.4)

 
    
    for i in range(0,2):
        lng = lam306_vip[i,:]
        lat = r306thet_vip[i,:]

        dl = lng[1:36] - lng[0:35]

        p_cond = np.where(abs(dl) > 180)[0]

        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white')
        
        lng = lam_vip[i,:]
        lat = rthet_vip[i,:]
        dl - lng[1:36] - lng[0:35]
        p_cond = np.where(abs(dl) > 180)[0]
        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white', linestyle='dashed')
        
        axes.set_xlabel('Longitude (Degrees)')
        axes.set_ylabel('Latitude (Degrees)')
    
    #folder = input('Enter file path of folder to save plot (map): ')
    #plt.savefig('%s_%s_map.png' % (obs_id,st))
    plt.savefig(str(folder) + '/%s_map%03i.png' %(obs_id,iteration))
    plt.close()
    #plt.savefig(r'%s_map%03i.png' % (obs_id,iteration))
    return

def plotprops_int_time(iteration,props,st,units,sup_lon_list,sup_lat_list,obs_id,folder):
    
    """Plots the exposure time of Chandra in seconds across Jupiter's surface on a S3 longitude vs. latitude grid. Multiple timemaps are produced depending on the number of intervals the obsrvation has baan split into. The Io and magnetotail footprints are also plotted. The inputs are defined as follows:
    
    iteration:      the counter used to map all time interavls of the Chandra observation.
    
    props:   the property of the X-rays plotted onto the map. For example, Chandra code provides a flux, exposure time and a brightness (using a conversion factor).
    
    st:     string for the title of the plot.
    
    units:  units of the color bar used in the map.
    
    sup_lon_list:   the list of longitudes for the X-ray emission found.
    
    sup_lat_list:   the list of latitudes for the X-ray emission found.
    
    obs_id:  the observation id - this is set as default
    
    folder:   the file path for the saved plots"""
        
    geomsy_data = scipy.io.readsav('geomsy.sav')
    lam_vip = geomsy_data['lam_vip'] 
    rthet_vip = geomsy_data['rthet_vip'] 
    rns_vip = geomsy_data['rns_vip'] 
    lam306_vip = geomsy_data['lam306_vip'] 
    r306thet_vip = geomsy_data['r306thet_vip'] 
    r306_vip = geomsy_data['r306_vip']
    
    fig, axes=plt.subplots(figsize=(12,8))
    axes = plt.gca()
    axes.set_xlim(360,0)
    axes.set_ylim(-90,90)
    axes.plot([0,359], [-90,90],linestyle='None')
    axes.set_title(st)
    axes.text(175, -115, 'Max = %f %s' % (np.amax(props), units), fontsize=15, horizontalalignment='center')

    lngarr = np.arange(360)

    
    for i in range(0,361,20):
        axes.plot([i,i], [-90,90], linestyle = 'dashed', color = 'white', alpha=0.4)
        
    for i in range(0,181,20):
        axes.plot([0,360], [i-90, i-90], linestyle='dashed', color='white', alpha=0.4)

 
    
    for i in range(0,2):
        lng = lam306_vip[i,:]
        lat = r306thet_vip[i,:]

        dl = lng[1:36] - lng[0:35]

        p_cond = np.where(abs(dl) > 180)[0]

        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white')
        
        lng = lam_vip[i,:]
        lat = rthet_vip[i,:]
        dl - lng[1:36] - lng[0:35]
        p_cond = np.where(abs(dl) > 180)[0]
        lat = np.roll(lat, -p_cond[0]-1)
        lng = np.roll(lng, -p_cond[0]-1)
        axes.plot(lng,lat, color='white', linestyle='dashed')
        
        
        axes.set_xlabel('Longitude (Degrees)')
        axes.set_ylabel('Latitude (Degrees)')
        
        
    max_cbar = np.amax(props)
    #max_cbar = 0.025
    mesh=axes.pcolormesh(np.arange(0,360), np.arange(-90,91), props.T, norm=colors.PowerNorm(gamma=0.75), vmin=0, vmax = max_cbar,\
                        alpha = 0.75, cmap = 'Blues')
    cbar = plt.colorbar(mesh)
    cbar.set_label('%s' % units, rotation=270, labelpad = 20)
    #axes.scatter(juno_radio_lon[iteration], juno_radio_lat[iteration], alpha=1, color='orange', label = 'Juno Waves (Radio)')#,\
                #edgecolors = 'black')
        
   
    axes.scatter(sup_lon_list[iteration], sup_lat_list[iteration] - 90, color='white', edgecolors = 'black', label='Chandra')
    #axes.legend(loc='center left', bbox_to_anchor=(1.22, 0.5))
    
    #folder = input('Enter file path of folder to save plot (timemap): ')
    #plt.savefig('%s_%s_map.png' % (obs_id,st))
    plt.savefig(str(folder) + '/%s_timemap%03i.png' %(obs_id,iteration))
    plt.close()
    #plt.savefig(r'%s_map%03i.png' % (obs_id,iteration))
    return


    
def datetime_range(start, end, delta):
    
    """Splits the Chandra observation into delta intervals where delta is given in minutes. The 'start' and 'end' times are the beginning and end of the observation converted from Chandra seconds to datetime format"""
    
    current = start
    while current < end:
        yield current
        current += delta
    

