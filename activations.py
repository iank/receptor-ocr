#!/usr/bin/env python
import receptor_common as rc
import sys

# Usage: activations.py directory/ labels.txt field.npy idx1 [idx2 .. [idxN]]

def tblwrap(x):
    if x > 0.8:
        color = "#00FF00"
    elif x > 0.15:
        color = "#ADD8E6"
    elif x > 0.001:
        color = "#FFE3EB"
    else:
        color = "#FFFFFF"

    return '<td style="padding: 0; margin: 0; background-color:' + color + '">&nbsp;</td>'

images = rc.load_images(sys.argv[1], sys.argv[2])
receptors = rc.load_field(sys.argv[3])
idxs = [int(x) for x in sys.argv[4:]]

for k in idxs:
    receptors[k] = rc.compute_activation(images, receptors[k])
idxs = sorted(idxs, key=lambda k: receptors[k]['HXY'], reverse=True)

print('<table style="border-collapse: collapse;">')
print("<tr><th>#</th><th>Hyx</th><th>Hxy</th><th>_</th><th>1</th><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th><th>F</th><th>G</th><th>H</th><th>I</th><th>J</th><th>K</th><th>L</th><th>M</th><th>N</th><th>O</th><th>P</th><th>Q</th><th>R</th><th>S</th><th>T</th><th>U</th><th>V</th><th>W</th><th>X</th><th>Y</th><th>Z</th></tr>")
for k in idxs:
    receptors[k] = rc.compute_activation(images, receptors[k])
    receptor = receptors[k]

    labels = sorted(receptor['p1_x'])
    p1x = " ".join([tblwrap(receptor['p1_x'][x]) for x in labels])

    print('<tr><td>{0}</td><td>{1:.2f}</td><td style="border-right: 1px solid">{2:.2f}</td>{3}</tr>'.format( k, 0+receptor['HYX'], 0+receptor['HXY'], p1x)) 

print("</table>")
