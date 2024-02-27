import mlx.core as mx


fn = lambda x, y: x @ y

a = mx.random.normal(shape=(10000, 10000))
b = mx.random.normal(shape=(10000, 10000))

mx.eval(fn(a, b))

# Memory: 1.13 GB

c = mx.random.normal(shape=(10000, 10000))
d = mx.random.normal(shape=(10000, 10000))

mx.eval(fn(c, d))

# Memory: 1.88 GB

a = None
b = None
c = None
d = None
y = None

e = mx.random.normal(shape=(10000, 10000))
f = mx.random.normal(shape=(10000, 10000))

mx.eval(fn(e, f))

# Memory: 1.88 GB

g = mx.random.normal(shape=(10000, 10000))
h = mx.random.normal(shape=(10000, 10000))

mx.eval(fn(g, h))

# Memory: 1.88 GB