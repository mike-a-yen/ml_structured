import hashlib

def compute_sha256(filename):
    with open(filename,'rb') as fo:
        return hashlib.sha256(fo.read()).hexdigest()
