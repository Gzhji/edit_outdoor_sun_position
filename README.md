# edit_outdoor_sun_position
This respository describes the process for editing the outdoor sun in the captured fisheye image


# Remove the Sun 

Use the sun theta (azi) and phi (alt) angle from [source](https://www.suncalc.org/#/27.6936,-97.5195,3/2024.01.23/16:05/1/3) and Building Orientaion measured from the room. 

run command: *python new_sun2circle.py --theta 194.87 --phi 40.31 --building_ori 264*

<img src="sun_mask.png" width="200" height="200"/>
<img src="IMG_0067.JPG" width="200" height="200"/>
The generated mask will be paired with original image to inpainting the sun and nearby region.


 
