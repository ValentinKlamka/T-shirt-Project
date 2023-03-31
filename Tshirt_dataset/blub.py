import os
import re
from pathlib import Path
os.chdir('../img_train/')
print (os.getcwd())
#print first 10 file names
print(os.listdir()[:10])
#save first file in variable
img_path=Path(os.listdir()[0])
#print everything before .png
id= re.split('\.+',img_path.name)[0]
print(id)
