import os
from html.parser import HTMLParser

parser = HTMLParser()

def maskSamples():
    masked_output = ""
    fPath_in = os.path.join(os.getcwd(), 'Authorship-Identification-Systems','target.txt')
    with open(fPath_in, encoding="utf-8") as f:
        raw_line = f.read()
        #masked_output.append(raw_line.upper())
        masked_output = raw_line.upper()
    
    fPath_out = os.path.join(os.getcwd(), 'Authorship-Identification-Systems','KentuckyMasked.txt')
    with open(fPath_out, 'w') as outfile:
        '''
        for line in masked_output:
            outfile.write('{}\n'.format(parser.unescape(line)))
        '''
        outfile.writelines('{}\n'.format(parser.unescape(masked_output)))

maskSamples()