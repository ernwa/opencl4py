opencl4py2
==========

**Minimalist OpenCL for python based on opencl4py**

_Why?_

Pyopencl is the de facto standard python bindings for OpenCL, but in the course of my research on real-time infrared projector systems I decided that it was lacking in certain essential features. The author seems to focus on complex and leaky high-level abstractions at the expense of reliable and intuitive access to the underlying OpenCL layer, and the interface code is compiled C++, which makes it hard to tinker with.

When I found opencl4py, I thought it would be the answer. A minimalist set of classes wrapping cffi bindings were the perfect architecture for research work. One important feature is that names and parameter orders mostly align intuitively with the OpenCL C documentation, which is really important for figuring out errors in CL code.

Unfortunately the original opencl4py is very lacking in completeness. It has no support for cl_image types, nor for cl/gl sharing, nor even barriers, so I added these things. Now that it's doing pretty much everything I need, it's time to share.


_Features Supported:_

This version has been extensively used on both Linux and MacOSX with Python 2.7. It supports:

- Buffer
- Image & Sampler
- Pipe & SVM (I haven't tested these)
- creating Context from OpenGL context
- creating Buffer & Image from OpenGL objects

Opencl4py itself was made to work for Python 3.3, Python 3.4 and PyPy as well as on Windows; I have not tested my additions on these platforms yet but it should work. GL/CL interop has been written to use WGL when Windows is present.


_Installation:_

To install directly from github run:
```bash
pip install git+https://github.com/ernwa/opencl4py2
```

To run the tests, execute:

for Python 2.7:
```bash
PYTHONPATH=src nosetests -w tests
```

for Python 3.3, 3.4:
```bash
PYTHONPATH=src nosetests3 -w tests
```

for PyPy:
```bash
PYTHONPATH=src pypy -m nose -w tests
```

_Example:_

```python
import opencl4py as cl
import logging
import numpy


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    platforms = cl.Platforms()
    logging.info("OpenCL devices available:\n\n%s\n",
                 platforms.dump_devices())
    ctx = platforms.create_some_context()
    queue = ctx.create_queue(ctx.devices[0])
    prg = ctx.create_program(
        """
        __kernel void test(__global const float *a, __global const float *b,
                           __global float *c, const float k) {
          size_t i = get_global_id(0);
          c[i] = (a[i] + b[i]) * k;
        }
        """)
    krn = prg.get_kernel("test")
    a = numpy.arange(1000000, dtype=numpy.float32)
    b = numpy.arange(1000000, dtype=numpy.float32)
    c = numpy.empty(1000000, dtype=numpy.float32)
    k = numpy.array([0.5], dtype=numpy.float32)
    a_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                              a)
    b_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,
                              b)
    c_buf = ctx.create_buffer(cl.CL_MEM_WRITE_ONLY | cl.CL_MEM_ALLOC_HOST_PTR,
                              size=c.nbytes)
    krn.set_args(a_buf, b_buf, c_buf, k[0:1])
    queue.execute_kernel(krn, [a.size], None)
    queue.read_buffer(c_buf, c)
    max_diff = numpy.fabs(c - (a + b) * k[0]).max()
    logging.info("max_diff = %.6f", max_diff)
```

Released under Simplified BSD License.
Copyright (c) 2014, Samsung Electronics Co.,Ltd.
improvements Copyright (c) 2017, Andrea Waite
