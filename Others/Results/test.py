#import numpy as np

#from pylatex import Document, Section, Subsection, Table, Math, TikZ, Axis, \
    #Plot
#from pylatex.numpy import Matrix
#from pylatex.utils import italic
#from pyx import *

##c = canvas.canvas()
##c.text(0, 0, "Hello, world!")
##c.stroke(path.line(0, 0, 2, 0))
##c.writeEPSfile()
##c.writePDFfile()


#doc = Document()
#section = Section('Yaay the first section, it can even be ' + italic('italic'))


#section.append('Some regular text')

#math = Subsection('Math that is incorrect', data=[Math(data=['2*3', '=', 9])])

#section.append(math)
#table = Table('rc|cl')
#table.add_hline()
#table.add_row((1, 2, 3, 4))
#table.add_hline(1, 2)
#table.add_empty_row()
#table.add_row((4, 5, 6, 7))


#table = Subsection('Table of something', data=[table])

#section.append(table)

#a = np.array([[100, 10, 20]]).T
#M = np.matrix([[2, 3, 4],
               #[0, 0, 1],
               #[0, 0, 2]])

#math = Math(data=[Matrix(M), Matrix(a), '=', Matrix(M*a)])
#equation = Subsection('Matrix equation', data=[math])

#section.append(equation)

#tikz = TikZ()

#axis = Axis(options='height=6cm, width=6cm, grid=major')

#plot1 = Plot(name='model', func='-x^5 - 242')
#coordinates = [
    #(-4.77778, 2027.60977),
    #(-3.55556, 347.84069),
    #(-2.33333, 22.58953),
    #(-1.11111, -493.50066),
    #(0.11111, 46.66082),
    #(1.33333, -205.56286),
    #(2.55556, -341.40638),
    #(3.77778, -1169.24780),
    #(5.00000, -3269.56775),
#]

#plot2 = Plot(name='estimate', coordinates=coordinates)

#axis.append(plot1)
#axis.append(plot2)

#tikz.append(axis)

#plot_section = Subsection('Random graph', data=[tikz])

#section.append(plot_section)

#doc.append(section)

#doc.generate_pdf()


def generate_pdf(pdfname,table):
    """
    Generates the pdf from string
    """
    import subprocess
    import os

    f = open('/Users/Mohsen/Documents/nlpy_mohsen/Numerical_Rezults/results.tex','w')
    tex = standalone_latex(table)   
    f.write(tex)
    f.close()

    proc=subprocess.Popen(['pdflatex','cover.tex'])
    subprocess.Popen(['pdflatex',tex])
    proc.communicate()
    os.unlink('cover.tex')
    os.unlink('cover.log')
    os.unlink('cover.aux')
    os.rename('cover.pdf',pdfname)
if __name__ == "__main__":
    import subprocess
    import os
    f = '/Users/Mohsen/Documents/nlpy_mohsen/Numerical_Rezults/results.tex'
    #output = subprocess.Popen(["pdflatex", 'results.tex'], stdout=subprocess.PIPE).communicate()[0]
    p = subprocess.Popen(["pdflatex " + '/Users/Mohsen/Documents/nlpy_mohsen/lsq/results.tex'], shell=True)
    sts = os.waitpid(p.pid, 0)[1]    
    print p    
    #subprocess.Popen(["pdflatex", f], shell=False)
    #return_value = subprocess.call(['pdflatex', f], shell=False)
    