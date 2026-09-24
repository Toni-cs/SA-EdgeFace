import urllib.request, socket
socket.setdefaulttimeout(15)
urls = [
    'https://raw.githubusercontent.com/wbiast/face-recognition/master/pairs.txt',
    'https://raw.githubusercontent.com/davidsandberg/facenet/master/src/lfw_eval/pairs.txt',
    'https://github.com/clcarwin/facenet-pytorch/raw/master/data/lfw/pairs.txt',
]
for url in urls:
    try:
        urllib.request.urlretrieve(url, 'datasets/pairs.txt')
        with open('datasets/pairs.txt') as f:
            first = f.readline().strip()
        print(f'OK from {url}: first line = {first}')
        break
    except Exception as e:
        print(f'FAIL {url}: {e}')