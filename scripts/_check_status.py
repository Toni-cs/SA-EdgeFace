import os
p = 'datasets/pairs.txt'
if os.path.exists(p):
    with open(p) as f:
        line1 = f.readline().strip()
    print(f'pairs.txt: first line = {line1[:50]}')
else:
    print('pairs.txt: NOT FOUND')

casia = 'datasets/raw/casia-webface'
if os.path.isdir(casia):
    dirs = os.listdir(casia)
    print(f'casia-webface: {len(dirs)} dirs, first: {dirs[:3]}')
else:
    print('casia-webface: NOT FOUND')

z = 'datasets/raw/casia-webface.7z'
if os.path.exists(z):
    print(f'7z: {os.path.getsize(z)/1e9:.2f} GB')