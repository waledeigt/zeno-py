# coding: utf-8

# ## <u> sso-freeze - Python </u>
# 
# As described in *Weigt et al. (in prep.)* The following code is a script that takes Gladstone's stop_hrci and translates it into python.
# This code has been split up into the different section for each different functions carried out.
# 
# <i>End Goal </i>: To simply enter the OBSid -> gather the selected files -> apply ''sso_freeze'' -> output the corrected file
# to a different text file
# 
# Sections are as follows:
# 
# *1)* Reading in the Chandra events file (uncorrected) and extraxt relevant header info </br>
# 
# *2)* Reading in the orbit empheris file and extract relevant header info </br>
# 
# *3)* Read in the ''chandra_horizons file'', extract relevant info and interpolate the data </br>
# 
# *4)* Apply the correction to allow the photons to be tracked to their position on Jupiter </br>
# 
# *5)* The new positions of the photons replace the uncorrected positions from the fits file and is written to a new fits file. </br>
# 

# #### <u><i> Hardwire locations </u></i> 
# 
# In the cells below, there are various hardwire locations that need to be inputted (hopefully once this code is properly optimised, there should be no need for this).
# 
# The hardwire locations are as follows:
# 
# <b>EDIT:</b>
# 
# *Section 1)*: <b>evt_location</b> -> enter path of the event file to be corrected and orbit ephemeris file
# 
# *Section 3)*: <b>eph_location</b> -> enter path of the chandra_horizons2000 file used (from folder of horizons2000 files) 
# 
# *Section 5)*: The save file will now be saved to the same location as the original event and orbit ephemeris file under the name "\hrcf%s_pytest_evt2.fits" where %s is the obs_id (automatically inputted when file is saved).

# In[2]:


#Purpose: Read in Chandra event file and empheris files. Correct event file by time-tagging photons to their position
#on Jupiter. New fits file should produce a projection of the x-rays on Jupiter.
#Category: Chandra fits file correction (Jupiter)
#Authors: Dale Weigt (D.M.Weigt@soton.ac.uk), apadpted from Randy Gladstone's 'stop_hrci' IDL script

"""All the relevant packages are imported for code below"""

import numpy as np
import pandas as pd
import os
from astropy.io import fits as pyfits
from scipy import interpolate

"""Below are a list of defined functions used in the code below
----------------------------------------------------------------"""


def doy_frac(DOY, hour, minutes, seconds): # function takes a given DOY, hour, minutes, seconds...
    frac = DOY + hour/24.0 + minutes/1440.0 + seconds/86400.0 #... and calualtes the correct DOY fraction.
    return frac

# AU to meter conversion - useful later on (probably a function built in already)
AU_2_m = 1.49598E+11


# In[3]:
ACIS = input('Are you correcting a fits file from the ACIS instrument? [y/n]: ') 

"""SECTION 1)"""

folder_path = input('Enter file path of event and orbit ephermis file: ')   

# The code below reads in the Chandra event file and extracts the relevant header info needed later
evt_location = []

if ACIS == 'y':
    for file in os.listdir(str(folder_path)):
        if file.startswith("acisf") and file.endswith("evt2.fits"):
            evt_location.append(os.path.join(str(folder_path), file))

else:    
    for file in os.listdir(str(folder_path)):
        if file.startswith("hrcf") and file.endswith("evt2.fits"):
            evt_location.append(os.path.join(str(folder_path), file))


evt_file = pyfits.open(evt_location[0])
# event file is read in and opened to extract header info
evt_hdr = evt_file[1].header # event file header...
evt_data = evt_file[1].data #...and associated date with each header
evt_time = evt_data['TIME'] # time of each photon observred as it reaches detector(???)
tstart = evt_hdr['TSTART'] # start time of observation (in seconds)
tend = evt_hdr['TSTOP'] # end time of observation (in seconds)
obs_id = evt_hdr['OBS_ID']
start_date = evt_hdr['DATE-OBS'] # start date of observation
evt_x = evt_data['X'] # x position of target
evt_y = evt_data['Y'] # y position of target
#end_time = evt_hdr['TSTOP']
RA_0 = evt_hdr['RA_NOM'] # RA origin of target at the start of the observation
DEC_0 = evt_hdr['DEC_NOM'] # DEC origin of target at the start of the observation
evt_file.close()

evt_date = pd.to_datetime(start_date) # converts the date to a timestamp - to allow the time to be separated and used for
# calculation of the DOY and DOYFRAC
evt_hour = evt_date.hour # hour of observation
evt_doy = evt_date.strftime('%j') # doy of observation
evt_mins = evt_date.minute # minute of observation
evt_secs = evt_date.second # second of observation
evt_DOYFRAC = doy_frac(float(evt_doy), float(evt_hour), float(evt_mins), float(evt_secs)) # calculating the DOYFRAC
chand_time = (evt_time - tstart)/86400.0 # calculating time cadence of chandra...
doy_chandra = chand_time + evt_DOYFRAC #... to calculate the DOY of chandra


# In[4]:


"""SECTION 2)"""
# The code below reads in the orbit empheris file for the identified OBSid

orb_location = []

for file in os.listdir(str(folder_path)):
    #if file.startswith("orbit") and file.endswith("eph0.fits"):    
    if file.startswith("orbit") and file.endswith("eph1.fits"):
        orb_location.append(os.path.join(str(folder_path), file))


    
# file used corresponding to the event file

orb_file = pyfits.open(orb_location[0])
# orbit empheris file is read in...
hdr = orb_file[1].header #...header information is extacted...
data = orb_file[1].data #...and the relevant data us also extracted
orb_time = data['TIME'] # time of observation when Jovian photons reach spaecraft 
orb_x = data['X'] # x position of spacecraft
orb_y = data['Y'] # y position of spacecraft
orb_z = data['Z'] # z position of spacecraft
orb_file.close()

doy_sc = (orb_time - tstart) /86400.0 + evt_DOYFRAC # doy of spacecraft


# In[6]:


"""SECTION 3)"""


"""Brad Sinos's horizons code to extract the ephemeris file"""

from astropy.time import Time                   #convert between different time coordinates
from astropy.time import TimeDelta              #add/subtract time intervals 
from astroquery.jplhorizons import Horizons     #automatically download ephemeris 

# The start and end times are taken from the horizons file.
tstart_eph=Time(tstart, format='cxcsec') 

dt = TimeDelta(0.125, format='jd') 
tstop_eph=Time(tend, format='cxcsec')

# Below sets the parameters of what observer (geocentric) the ephemeris file is generated form. For example, '500' = centre of the Earth, '500@-151' = CXO
obj = Horizons(id=599,location='500',epochs={'start':(tstart_eph).iso, 'stop':(tstop_eph+dt).iso, 'step':'5m'}) # step size of ephemeris is set to 5-mins
eph_jup = obj.ephemerides()

# Extracts relevent date/time information needed from ephermeris file

eph_dates = pd.to_datetime(eph_jup['datetime_str']) 
eph_dates = pd.DatetimeIndex(eph_dates)
eph_doy = np.array(eph_dates.strftime('%j')).astype(int)
eph_hours = eph_dates.hour
eph_minutes = eph_dates.minute
eph_seconds = eph_dates.second

eph_doyfrac = doy_frac(eph_doy, eph_hours, eph_minutes, eph_seconds) # DOY fraction from ephermeris data

"""SECTION 4)"""

interpfunc_x = interpolate.interp1d(doy_sc, orb_x,fill_value="extrapolate")
interpfunc_y = interpolate.interp1d(doy_sc, orb_y,fill_value="extrapolate")
interpfunc_z = interpolate.interp1d(doy_sc, orb_z,fill_value="extrapolate")

# Above code creates a linear intepolation function that interpolates the spacecraft DOY and the positional coordinates from
# the orbital empheris file to...

orb_x_interp = interpfunc_x(eph_doyfrac)   
orb_y_interp = interpfunc_y(eph_doyfrac)
orb_z_interp = interpfunc_z(eph_doyfrac)

#... the DOY from the empheris file (2 day window in this case) to produce the new orbit positional coordinates within this time
#range
r_jup = np.array(eph_jup['delta'].astype(float))*AU_2_m # Jupiter chandra distance 
dec_jup = np.deg2rad(np.array(eph_jup['DEC'].astype(float))) # DEC of Jupiter during observation
ra_jup = np.deg2rad(np.array(eph_jup['RA'].astype(float)))# RA of Jupiter duting observation
xp = (r_jup * np.cos(dec_jup) * np.cos(ra_jup)) - orb_x_interp
yp = (r_jup * np.cos(dec_jup) * np.sin(ra_jup)) - orb_y_interp
zp = (r_jup * np.sin(dec_jup)) - orb_z_interp
# Above is (x,y,z) position of Jupiter during obsrvation and...
rp = np.sqrt(xp**2 + yp**2 + zp**2)
#... the absoulte value of the Jupiter_Earth radius vector
rap_jup = (np.rad2deg(np.arctan2(yp,xp)) + 720.0) % 360  # RA of Jupiter at the observed coordinates
decp_jup = (np.rad2deg(np.arcsin(zp/rp))) # DEC of Jupiter at the observed coordinates

cc = np.cos(np.deg2rad(DEC_0)) # offset from Jupiter to allow photons to be tracked(?)

interpfunc_ra_jup = interpolate.interp1d(eph_doyfrac, rap_jup) 
interpfunc_dec_jup = interpolate.interp1d(eph_doyfrac, decp_jup) 
ra_jup_interp = interpfunc_ra_jup(doy_chandra) # interpolated RA of Jupiter and the DOY of the emphermis file to the Chandra DOY
dec_jup_interp = interpfunc_dec_jup(doy_chandra) # interpolated DEC of Jupiter and the DOY from the emphermis file to the Chandra
#DOY

if ACIS == 'y':
    scale = 0.4920 # units of pixels/arcsec
    xx = (evt_x - (RA_0 - ra_jup_interp) * 3600.0 / scale * cc).astype(float) # corrected x position of photons
    yy = (evt_y + (DEC_0 - dec_jup_interp) * 3600.0 / scale).astype(float) # corrected y position of photons

else:
    scale = 0.13175 # untis of pixels/arcsec 
    xx = (evt_x - (RA_0 - ra_jup_interp) * 3600.0 / scale * cc).astype(float) # corrected x position of photons
    yy = (evt_y + (DEC_0 - dec_jup_interp) * 3600.0 / scale).astype(float) # corrected y position of photons
    

"""SECTION 5)"""
if ACIS == 'y':
    new_evt_location = (str(folder_path) + f"/acisf{obs_id}_pytest_evt2.fits") # path of the location
    # for the corrected fits file (with the photons corrected for the position).
else:
    new_evt_location = (str(folder_path) + f"/hrcf{obs_id}_pytest_evt2.fits") # path of the location
    # for the corrected fits file (with the photons corrected for the position)

new_evt_data, new_evt_header = pyfits.getdata(evt_location[0], header=True)
# original fits file is read in again to obtain the data - data and header values are assigned to a new variable to avoid overwriting
# the original file
new_evt_data['X'] = xx # New x coord of photons added to fits file under previous data header 'X'
new_evt_data['Y'] = yy # New y coord of photons added to fits file under previous data header 'Y'
pyfits.writeto(new_evt_location, new_evt_data, new_evt_header, overwrite=True)
# new fits file is written with the correction to the position of the photons witht he original file remaining the same
#pyfits.close()
