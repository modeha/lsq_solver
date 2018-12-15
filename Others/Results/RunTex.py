import subprocess
import os 
f = os.getcwd()+'/'
p = "/usr/texbin/pdflatex  "
print [p + f+'results.tex']
subprocess.Popen([p + f+'results.tex'], shell=True)
#sts = os.waitpid(p.pid, 0)[1] 
h = 'open '+f+'results.pdf'
subprocess.Popen(['open '+f+'results.pdf'],shell=True)    

    