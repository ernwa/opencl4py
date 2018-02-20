from __future__ import print_function
"""
Copyright (c) 2014, Samsung Electronics Co.,Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of Samsung Electronics Co.,Ltd..
"""

"""
opencl4py - OpenCL cffi bindings and helper classes.
URL: https://github.com/Samsung/opencl4py
Original author: Alexey Kazantsev <a.kazantsev@samsung.com>
"""

"""
Helper classes for OpenCL cffi bindings.
"""
import opencl4py._cffi as cl
import sys
import numpy as np


CL_TYPE_FROM_DTYPE = {
                           np.float16: 'half',    np.float32: 'float',
	np.int8:    'char',    np.int16:   'short',   np.int32:   'int',
	np.uint8:   'uchar',   np.uint16:  'ushort',  np.uint32:  'uint'
}


def dtype_cl_type_name(dtype):
    return CL_TYPE_FROM_DTYPE[ np.dtype(dtype).type ]


def ensure_type(typespec, obj, cast=False):
    if not isinstance(obj, cl.ffi.CData):  # cffi object
        return cl.ffi.new(typespec, obj)
    if cast:
        return cl.ffi.cast(typespec, obj)  # try to cast it
    return obj


def typeof_function(ffi, name):   # without function needing to be in lib
    with ffi._lock:
        tp, _ = ffi._parser._declarations['function ' + name]
        BType = ffi._get_cached_btype(tp)
    return BType

def declared_types(ffi, typename='function'):
    return [k for k in ffi._parser_declarations if typename in k]


class Extensions(object):
    def __init__(self, ffi):
        self.ffi = ffi
        self.extension = {}
        self.functions = {}

    def __getattr__(self, funcname):
        return self.functions[funcname]

    def register(self, extension_name, extension_source):
        with cl.lock:
            prev_declared = set(declared_types(self.ffi))
            self.ffi.cdef(extension_source)
            newly_declared = set(declared_types(self.ffi)) - prev_declared

        declared_funcs = [ n.split()[1] for n in newly_declared ]   # remove "function"

        self.source[extension_name] = (declared_funcs, extension_source)
        for nf in declared_funcs:
            print( nf )
#            self.functions[nf] =






def lookup_const(name):
    return cl.__dict__[name]


def name_lookup_table(*names):
    return { cl.__dict__[name]: name for name in names }


def register_extension(name, c_prototypes):
    cl.extensions[name] = c_prototypes
    cl.cdef(c_prototypes)


def check_error(err, funcname, extra_info=''):
    if err:
        raise CLRuntimeError(
            "%s() failed with error %s%s" %
            ( funcname, CL.get_error_description(err), extra_info), err )


def get_wait_list( wait_for ):
    """Returns cffi event list and number of events
    from list of Event objects, returns (None, 0) if wait_for is None.
    """
    if wait_for:
        n_events = len(wait_for)
        wait_list = cl.ffi.new("cl_event[]", [ev.handle for ev in wait_for] )
    else:
        n_events = 0
        wait_list = cl.ffi.NULL
    return (wait_list, n_events)


def wait_for_events( events, lib=None ):
    """Wait on list of Event objects.
    """
    lib = lib or cl.lib
    wait_list, n_events = get_wait_list(events)
    n = lib.clWaitForEvents(n_events, wait_list)
    check_error(n, 'clWaitForEvents')


class CLRuntimeError(RuntimeError):
    def __init__(self, msg, code):
        super(CLRuntimeError, self).__init__(msg)
        self.code = code


class CL(object):
    """Base OpenCL class.

    Attributes:
        _lib: handle to cffi.FFI object.
        _handle: cffi handle to OpenCL object.
    """

    ERRORS = name_lookup_table(
        "CL_SUCCESS", "CL_DEVICE_NOT_FOUND", "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE", "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY", "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP", "CL_IMAGE_FORMAT_MISMATCH", "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE", "CL_MAP_FAILURE", "CL_MISALIGNED_SUB_BUFFER_OFFSET",
        "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", "CL_COMPILE_PROGRAM_FAILURE",
        "CL_LINKER_NOT_AVAILABLE", "CL_LINK_PROGRAM_FAILURE", "CL_DEVICE_PARTITION_FAILED",
        "CL_KERNEL_ARG_INFO_NOT_AVAILABLE", "CL_INVALID_VALUE", "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM", "CL_INVALID_DEVICE", "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES", "CL_INVALID_COMMAND_QUEUE","CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT", "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE","CL_INVALID_SAMPLER", "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS", "CL_INVALID_PROGRAM", "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME", "CL_INVALID_KERNEL_DEFINITION", "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX", "CL_INVALID_ARG_VALUE", "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS", "CL_INVALID_WORK_DIMENSION", "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE", "CL_INVALID_GLOBAL_OFFSET", "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT", "CL_INVALID_OPERATION", "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE","CL_INVALID_MIP_LEVEL", "CL_INVALID_GLOBAL_WORK_SIZE",
        "CL_INVALID_PROPERTY", "CL_INVALID_IMAGE_DESCRIPTOR", "CL_INVALID_COMPILER_OPTIONS",
        "CL_INVALID_LINKER_OPTIONS", "CL_INVALID_DEVICE_PARTITION_COUNT",
        "CL_INVALID_PIPE_SIZE", "CL_INVALID_DEVICE_QUEUE",
        "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR" )

    if sys.platform == 'darwin':
        ERRORS[cl.CL_INVALID_GL_CONTEXT_APPLE] = "CL_INVALID_GL_CONTEXT_APPLE"


    CHANNEL_ORDERS = name_lookup_table(
        "CL_R", "CL_A", "CL_RG", "CL_RA", "CL_RGB", "CL_RGBA", "CL_BGRA", "CL_ARGB",
        "CL_INTENSITY", "CL_LUMINANCE", "CL_Rx", "CL_RGx", "CL_RGBx", "CL_DEPTH",
        "CL_DEPTH_STENCIL", "CL_sRGB", "CL_sRGBx", "CL_sRGBA", "CL_sBGRA", "CL_ABGR",
        "CL_NV21_IMG", "CL_YV12_IMG", "CL_1RGB_APPLE", "CL_BGR1_APPLE",
        "CL_YCbYCr_APPLE", "CL_CbYCrY_APPLE", "CL_ABGR_APPLE" )

    CHANNEL_TYPES = name_lookup_table(
        "CL_SNORM_INT8", "CL_SNORM_INT16", "CL_UNORM_INT8", "CL_UNORM_INT16",
        "CL_UNORM_SHORT_565", "CL_UNORM_SHORT_555", "CL_UNORM_INT_101010",
        "CL_SIGNED_INT8", "CL_SIGNED_INT16", "CL_SIGNED_INT32",
        "CL_UNSIGNED_INT8", "CL_UNSIGNED_INT16", "CL_UNSIGNED_INT32",
        "CL_HALF_FLOAT", "CL_FLOAT","CL_UNORM_INT24", "CL_UNORM_INT_101010_2",
        "CL_SFIXED14_APPLE", "CL_BIASED_HALF_APPLE" )

    GL_OBJECT_TYPES = name_lookup_table(
        "CL_GL_OBJECT_BUFFER", "CL_GL_OBJECT_TEXTURE2D",
        "CL_GL_OBJECT_TEXTURE3D", "CL_GL_OBJECT_RENDERBUFFER",
        "CL_GL_OBJECT_TEXTURE2D_ARRAY", "CL_GL_OBJECT_TEXTURE1D",
        "CL_GL_OBJECT_TEXTURE1D_ARRAY", "CL_GL_OBJECT_TEXTURE_BUFFER" )

    CL_OBJECT_TYPES = name_lookup_table(
        "CL_MEM_OBJECT_BUFFER", "CL_MEM_OBJECT_IMAGE2D", "CL_MEM_OBJECT_IMAGE3D",
        "CL_MEM_OBJECT_IMAGE2D_ARRAY", "CL_MEM_OBJECT_IMAGE1D", "CL_MEM_OBJECT_IMAGE1D_ARRAY",
        "CL_MEM_OBJECT_IMAGE1D_BUFFER", "CL_MEM_OBJECT_PIPE" )

    def __init__(self):
        self._lib = cl.lib  # to hold the reference
        self._gllib = cl.gllib
        self._handle = None
        self.from_gl = False

    def check_error(self, err, funcname, extra_info='', release=False):
        if err:
            if release and hasattr(self, "_handle"):
                self._handle = None
#                self._del_ref(self)     # shouldn't these both be here?q
            raise CLRuntimeError(
                "%s() failed with error %s%s" %
                ( funcname, CL.get_error_description(err), extra_info), err )


    def __eq__(self, other):
            if isinstance(other, type(self)):
                return self.handle == other.handle
            return NotImplemented

    def __ne__(self, other):
            result = self.__eq__(other)
            if result is NotImplemented:
                return result
            return not result

    @property
    def handle(self):
        """Returns cffi handle to OpenCL object.
        """
        return self._handle

    @staticmethod
    def extract_ptr_and_size(host_array, size):
        """Returns cffi pointer to host_array and its size.
        """
        if hasattr(host_array, "__array_interface__"):
            host_ptr = host_array.__array_interface__["data"][0]
            if size is None:
                size = host_array.nbytes
        else:
            host_ptr = host_array
            if size is None:
                raise ValueError("size should be set "
                                 "in case of non-numpy host_array")
        return (cl.ffi.NULL if host_ptr is None
                else cl.ffi.cast("void*", host_ptr), size)


    @staticmethod
    def extract_ptr(host_array):
        if hasattr(host_array, "__array_interface__"):
            host_ptr = host_array.__array_interface__["data"][0]
        else:
            host_ptr = host_array
        return cl.ffi.NULL if host_ptr is None else cl.ffi.cast("void*", host_ptr)


    @staticmethod
    def get_wait_list(wait_for):
        return get_wait_list(wait_for)

    @staticmethod
    def get_error_name_from_code(code):
        return CL.ERRORS.get(code, "UNKNOWN")

    @staticmethod
    def get_error_description(code):
        return "%s (%d)" % (CL.get_error_name_from_code(code), code)


class Sampler(CL):  # TODO: test, implement clCreateSamplerWithProperties
    def __init__(self, context, normalized_coords, addressing_mode, filter_mode ):
        """
        Creates a cl sampler object.

        Parameters:
            context:            Context object

            normalized_coords:  determines if the image coordinates specified
                                    are normalized (True, False)

            addressing_mode:    specifies how out-of-range image coordinates are
                                    handled when reading from an image
                                    (CL_ADDRESS_MIRRORED_REPEAT, CL_ADDRESS_REPEAT,
                                     CL_ADDRESS_CLAMP_TO_EDGE, CL_ADDRESS_CLAMP,
                                     CL_ADDRESS_NONE)

            filter_mode:        specifies the type of filter that must be applied
                                    when reading an image.
                                    (CL_FILTER_NEAREST, CL_FILTER_LINEAR)

        Returns:

        """
        super(Context, self).__init__()
        self.context = context
        normalized_coords = cl.CL_TRUE if normalized_coords else cl.CL_FALSE
        errcode_ret = cl.ffi.new('cl_int *')
        self.sampler = self.lib.clCreateSampler(  context.handle,
                                        normalized_coords,
                                        addressing_mode,
                                        filter_mode,
                                        errcode_ret )
        self.check_error(errcode_ret, 'clCreateSampler', release=True)

    def retain(self):
        errcode = self.lib.clRetainSampler( self.sampler )
        self.check_error(errcode, 'clRetainSampler')

    def release(self):
        errcode = self.lib.clReleaseSampler( self.sampler )
        self.check_error(errcode, 'clRetainSampler')

    def _get_sampler_info(self, param_name, buf ):
        sz = cl.ffi.new("size_t *")
        param_value_size = cl.ffi.sizeof(buf) if buf else 0
        errcode = self.lib.clGetSamplerInfo( self.sampler,
                                             param_name, param_value_size,
                                             buf, sz )
        self.check_error(errcode, 'clGetSamplerInfo')
        return sz

    @property
    def reference_count(self):
        count = cl.ffi.new('cl_uint')
        self._get_sampler_info( cl.CL_SAMPLER_REFERENCE_COUNT, count)
        return count

    @property
    def context(self):
        try:
            return self._context
        except AttributeError:
            clctx = cl.ffi.new('cl_context')
            self._get_sampler_info( cl.CL_SAMPLER_CONTEXT, clctx)
            self._context = Context.from_cl_context( clctx )
            return self._context

    # TODO: fix these too
    # @property
    # def normalized_coords(self):
    #     return self._get_sampler_info( cl.CL_SAMPLER_NORMALIZED_COORDS, 'cl_bool')
    #
    # @property
    # def addressing_mode(self):
    #     return self._get_sampler_info( cl.CL_SAMPLER_ADDRESSING_MODE, 'cl_addressing_mode')
    #
    # @property
    # def filter_mode(self):
    #     return self._get_sampler_info( cl.CL_SAMPLER_FILTER_MODE, 'cl_filter_mode')


class Event(CL):
    """Holds OpenCL event.

    Attributes:
        profiling_values:
            dictionary of profiling values
            if get_profiling_info was ever called;
            keys: CL_PROFILING_COMMAND_QUEUED,
                  CL_PROFILING_COMMAND_SUBMIT,
                  CL_PROFILING_COMMAND_START,
                  CL_PROFILING_COMMAND_END;
            values: the current device time counter in seconds (float),
                    or 0 if there was an error, in such case, corresponding
                    profile_errors will be set with the error code.
        profiling_errors: dictionary of profiling errors
                          if get_profiling_info was ever called.
    """
    def __init__(self, handle):
        super(Event, self).__init__()
        self._handle = handle

    @classmethod
    def from_gl_sync(cls, context, sync ):
        """Creates an event object linked to an OpenGL sync object.

        Parameters:
            context: A Context created from an OpenGL context or share group.
            sync: The name of a sync object in the associated GL share group.
        Returns:
            Event object.
        """
        return context.create_event_from_gl_sync( sync )

    @staticmethod
    def wait_multi(wait_for, lib=None):
        wait_for_events( wait_for, lib=lib )

    def wait(self):
        """Waits on this event.
        """
        Event.wait_multi((self,), self._lib)


    def get_profiling_info(self, raise_exception=True):
        """Get profiling info of the event.

        Queue should be created with CL_QUEUE_PROFILING_ENABLE flag,
        and event should be in complete state (wait completed).

        Parameters:
            raise_exception: raise exception on error or not,
                             self.profiling_values, self.profiling_errors
                             will be available anyway.

        Returns:
            tuple of (profiling_values, profiling_errors).
        """
        vle = cl.ffi.new("cl_ulong[]", 1)
        sz = cl.ffi.sizeof(vle)
        vles = {}
        errs = {}
        for name in (cl.CL_PROFILING_COMMAND_QUEUED,
                     cl.CL_PROFILING_COMMAND_SUBMIT,
                     cl.CL_PROFILING_COMMAND_START,
                     cl.CL_PROFILING_COMMAND_END):
            vle[0] = 0
            n = self._lib.clGetEventProfilingInfo(
                self.handle, name, sz, vle, cl.ffi.NULL)
            vles[name] = 1.0e-9 * vle[0] if not n else 0.0
            errs[name] = n
        self.profiling_values = vles
        self.profiling_errors = errs
        if raise_exception:
            for err in errs.values():
                self.check_error(err, 'clGetEventProfilingInfo')
        return (vles, errs)

    @property
    def command_execution_status(self):
        param_value = cl.ffi.new('cl_int*')
        self._get_event_info( cl.CL_EVENT_COMMAND_EXECUTION_STATUS, param_value )
        return int(param_value)

    @property
    def reference_count(self):
        param_value = cl.ffi.new('cl_uint*')
        self._get_event_info( cl.CL_EVENT_REFERENCE_COUNT, param_value )
        return int(param_value)

    @property
    def command_type(self):
        param_value = cl.ffi.new('cl_command_type*')
        self._get_event_info( cl.CL_EVENT_COMMAND_TYPE, param_value )
        return int(param_value)

### foo
    @property
    def queue(self):
        try:
            return self._queue
        except AttributeError:
            handle = self._get_event_info( cl.CL_EVENT_CONTEXT, 'cl_context' )
            self._queue = Queue.from_cl_context( handle )
            return self._queue


    def _get_event_info(self, param_name, param_value):
        param_value_size_ret = cl.ffi.new('size_t*')
        errcode = self.lib.clGetEventInfo( self.handle,
                                           param_name, cl.ffi.sizeof(param_value),
                                           param_value, param_value_size_ret )
        self.check_error(errcode, 'clGetEventInfo')
        return param_value_size_ret[0]


    def _release(self):
        if self.handle is not None:
            self._lib.clReleaseEvent(self.handle)
            self._handle = None

    def __del__(self):
        self._release()


class Queue(CL):
    """Holds OpenCL command queue.

    Attributes:
        context: context associated with this queue.
        device: device associated with this queue.
    """

    def __init__(self, context, device, flags, properties=None):
        """Creates the OpenCL command queue associated with the given device.

        Parameters:
            context: Context instance.
            device: Device instance.
            flags: flags for the command queue creation.
            properties: dictionary of the OpenCL 2.0 queue properties.
        """
        super(Queue, self).__init__()
        context._add_ref(self)
        self._context = context
        self._device = device

        err = cl.ffi.new("cl_int *")
        if properties is None or device.version < 2.0:
            fnme = "clCreateCommandQueue"
            self._handle = self._lib.clCreateCommandQueue(
                context.handle, device.handle, flags, err)
        else:
            fnme = "clCreateCommandQueueWithProperties"
            if properties is None and flags == 0:
                props = cl.ffi.NULL
            else:
                if cl.CL_QUEUE_PROPERTIES not in properties and flags != 0:
                    properties[cl.CL_QUEUE_PROPERTIES] = flags
                props = cl.ffi.new("uint64_t[]", len(properties) * 2 + 1)
                for i, kv in enumerate(sorted(properties.items())):
                    props[i * 2] = kv[0]
                    props[i * 2 + 1] = kv[1]
            self._handle = self._lib.clCreateCommandQueueWithProperties(
                context.handle, device.handle, props, err)
        self.check_error(err[0], fnme, release=True)


    @classmethod
    def from_cl_queue(cls, queue_id):
        self = cls.__new__(cls)
        self._init_empty()
        assert self.device         # populate device list
        self._handle = queue_id

    def _get_command_queue_info(self, code, buf):
        sz = cl.ffi.new("size_t *")
        err = self._lib.clGetCommandQueueInfo(
            self.handle, code, cl.ffi.sizeof(buf), buf, sz)
        self.check_error(err, "clGetCommandQueueInfo")
        return sz[0]

    @property
    def context(self):
        """
        context associated with this queue.
        """
        try:
            return self._context
        except AttributeError:
            context_id = cl.ffi.new('cl_context*')
            self._get_command_queue_info(self.handle, context_id)
            self._context = Context.from_cl_context(context_id)
            return self._context


    @property
    def device(self):
        """
        device associated with this queue.
        """
        try:
            return self._device
        except AttributeError:
            device_id = cl.ffi.new('cl_device_id*')
            self._get_command_queue_info(self.handle, device_id)
            self._device = Device(device_id)
            return self._device


    def execute_kernel(self, kernel, global_size, local_size=None,
                       global_offset=None, wait_for=None, need_event=False):
        """Executes OpenCL kernel (calls clEnqueueNDRangeKernel).

        Parameters:
            kernel: Kernel object.
            global_size: global size.
            local_size: local size.
            global_offset: global offset.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        n_dims = len(global_size)
        global_work_size = cl.ffi.new("size_t[]", list(global_size))

        if local_size is None:
            local_work_size = cl.ffi.NULL
        else:
            if len(local_size) != n_dims:
                raise ValueError("local_size should be the same length "
                                 "as global_size")
            local_work_size = cl.ffi.new("size_t[]", list(local_size))

        if global_offset is None:
            global_work_offset = cl.ffi.NULL
        else:
            if len(global_work_offset) != n_dims:
                raise ValueError("global_offset should be the same length "
                                 "as global_size")
            global_work_offset = cl.ffi.new("size_t[]", list(global_offset))

        n = self._lib.clEnqueueNDRangeKernel(
                self.handle, kernel.handle,
                n_dims, global_work_offset, global_work_size, local_work_size,
                n_events, wait_list, event)

        self.check_error(n, "clEnqueueNDRangeKernel")

        return Event(event[0]) if event != cl.ffi.NULL else None


    def map_image(self, image, flags, region=None,
                   origin=None, blocking=True,
                   wait_for=None, need_event=False):
        """Maps image.

        Parameters:
            image: Image object.
            flags: mapping flags.
            region: (x,y,z) size of mapped image region.
            blocking: if the call would block until completion.
            origin: (x,y,z) offset location of mapped image region.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            (event, ptr, image_row_pitch, image_slice_pitch):
                    event - Event object or None if need_event == False,
                    ptr - pointer to the mapped image
                                (cffi void* converted to int).
                    image_row_pitch - scan-line pitch in bytes.
                    image_slice_pitch - size in bytes of each 2D slice.
        """
        if region is None:
            region = (image.width or 1, image.height or 1, image.depth or 1)
        elif len(region) == 2:
            region = tuple(region) + (1,)

        if origin is None:
            origin = (0, 0, 0)
        elif len(origin) == 2:
            origin = tuple(origin) + (0,)

        assert len(origin) == len(region)
        origin_struct = cl.ffi.new("size_t[3]", origin)
        region_struct = cl.ffi.new("size_t[3]", region)

        row_pitch = cl.ffi.new("size_t *")
        slice_pitch = cl.ffi.new("size_t *")

        err = cl.ffi.new("cl_int *")
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        ptr = self._lib.clEnqueueMapImage(
            self.handle, image.handle, blocking, flags, origin_struct, region_struct,
            row_pitch, slice_pitch, n_events, wait_list, event, err)

        self.check_error(err[0], "clEnqueueMapImage")

        return (None if event == cl.ffi.NULL else Event(event[0]),
                int(cl.ffi.cast("size_t", ptr)),
                int(row_pitch[0]), int(slice_pitch[0]))


    def map_buffer(self, buf, flags=cl.CL_MEM_READ_WRITE, size=None, offset=0,
                   blocking=True, wait_for=None, need_event=False):
        """Maps buffer.

        Parameters:
            buf: Buffer object.
            flags: mapping flags.
            size: mapping size.
            blocking: if the call would block until completion.
            offset: mapping offset.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            (event, ptr): event - Event object or None if need_event == False,
                          ptr - pointer to the mapped buffer
                                (cffi void* converted to int).
        """
        err = cl.ffi.new("cl_int *")
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        ptr = self._lib.clEnqueueMapBuffer(
                self.handle, buf.handle, blocking, flags, offset,
                size or (buf.size - offset), n_events, wait_list, event, err)

        self.check_error(err[0], "clEnqueueMapBuffer")

        return (None if event == cl.ffi.NULL else Event(event[0]),
                int(cl.ffi.cast("size_t", ptr)))


    def unmap(self, obj, ptr, wait_for=None, need_event=False):
        """Unmaps previously mapped object.

        Parameters:
            obj: Buffer object to unmap.
            ptr: pointer to the mapped object.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        ptr = CL.extract_ptr(ptr)
        wait_list, n_events = CL.get_wait_list(wait_for)
        n = self._lib.clEnqueueUnmapMemObject(
                self.handle, obj.handle, cl.ffi.cast("void*", ptr),
                n_events, wait_list, event)
        self.check_error(n, "clEnqueueUnmapMemObject")
        return Event(event[0]) if event != cl.ffi.NULL else None

    unmap_buffer = unmap
    unmap_image = unmap


    def read_buffer(self, buf, host_array, size=None, offset=0,
                    blocking=True, wait_for=None, need_event=False):
        """Copies from device buffer to host buffer.

        Parameters:
            buf: Buffer object.
            host_array: numpy array.
            blocking: if the read is blocking.
            size: size in bytes to copy (None for entire numpy array).
            offset: offset in the device buffer.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        host_ptr, size = CL.extract_ptr_and_size(host_array, size)
        n = self._lib.clEnqueueReadBuffer(
                self.handle, buf.handle, blocking, offset, size, host_ptr,
                n_events, wait_list, event)
        self.check_error(n, "clEnqueueReadBuffer")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def read_image(self, image, host_array, region=None, origin=(0,0),
                  src_row_pitch=0, src_slice_pitch=0,
                   blocking=True, wait_for=None, need_event=False):
        """Copies from device image to host buffer.

        Parameters:
            image: Image object.
            host_array: numpy array. shape will be used.
            blocking: if the call would block until completion.
            region: (x,y,z) size of read image region.
            origin: (x,y,z) offset of read image region.
            src_row_pitch: bytes between y slices (rows).
            src_slice_pitch: bytes between z slices.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False,
        """
        host_ptr = CL.extract_ptr(host_array)

        if region is None:
            region = [ max(x, 1) for x in (
                image.width,
                image.height,
                image.depth) ]

        region = tuple(region) + (3-len(region)) * (1,)
        origin = tuple(origin) + (3-len(origin)) * (0,)

        origin_struct = cl.ffi.new("size_t[3]", origin)
        region_struct = cl.ffi.new("size_t[3]", region)
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        n = self._lib.clEnqueueReadImage(
                self.handle, image.handle, blocking, origin_struct, region_struct,
                src_row_pitch, src_slice_pitch, host_ptr, n_events, wait_list, event)

        self.check_error(n, "clEnqueueReadImage")

        return None if event == cl.ffi.NULL else Event(event[0])


    def write_buffer(self, buf, host_array, size=None, offset=0,
                     blocking=True, wait_for=None, need_event=False):
        """Copies from host buffer to device buffer.

        Parameters:
            buf: Buffer object.
            host_array: numpy array.
            blocking: if the read is blocking.
            size: size in bytes to copy (None for entire numpy array).
            offset: offset in the device buffer.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        host_ptr, size = CL.extract_ptr_and_size(host_array, size)
        n = self._lib.clEnqueueWriteBuffer(
                self.handle, buf.handle, blocking, offset, size, host_ptr,
                n_events, wait_list, event)
        self.check_error(n, "clEnqueueWriteBuffer")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def write_image(self, image, host_array,
                   region=None, origin=(0,0),
                   src_row_pitch=0, src_slice_pitch=0,
                   blocking=True, wait_for=None, need_event=False):
        """Copies from host buffer to device image.

        Parameters:
            image: Image object.
            host_array: numpy array. shape will be used.
            blocking: if the call would block until completion.
            region: (x,y,z) size of written image region.
            origin: (x,y,z) offset of written image region.
            src_row_pitch: bytes between y slices (rows).
            src_slice_pitch: bytes between z slices.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False,
        """
        host_ptr = CL.extract_ptr(host_array)

        if region is None:
            region = [ max(x, 1) for x in (
                image.width,
                image.height,
                image.depth) ]

        region = tuple(region) + (3-len(region)) * (1,)
        origin = tuple(origin) + (3-len(origin)) * (0,)

        origin_struct = cl.ffi.new("size_t[3]", origin)
        region_struct = cl.ffi.new("size_t[3]", region)

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        n = self._lib.clEnqueueWriteImage(
                self.handle, image.handle, blocking, origin_struct, region_struct,
                src_row_pitch, src_slice_pitch, host_ptr, n_events, wait_list, event)

        self.check_error(n, "clEnqueueWriteImage")

        return None if event == cl.ffi.NULL else Event(event[0])


    def fill_image(self, image, fill_color,
                   origin=(0,0), region=None,
                   wait_for=None, need_event=False):
        """Enqueues a command to fill a region of an Image.

        Parameters:
            image: Image object.
            fill_color: numpy array that matches image format.
            origin: (x,y,z) offset of image region to fill.
            region: (x,y,z) size of image region to fill.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False,
        """
        ptr_pattern = CL.extract_ptr(fill_color)

        if region is None:
            region = [ max(x, 1) for x in (
                image.width,
                image.height,
                image.depth) ]

        region = tuple(region) + (3-len(region)) * (1,)
        origin = tuple(origin) + (3-len(origin)) * (0,)

        origin_struct = cl.ffi.new("size_t[3]", origin)
        region_struct = cl.ffi.new("size_t[3]", region)

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        n = self._lib.clEnqueueFillImage(
                self.handle, image.handle, ptr_pattern, origin_struct,
                region_struct, n_events, wait_list, event)

        self.check_error(n, "clEnqueueFillImage")

        return None if event == cl.ffi.NULL else Event(event[0])



    def copy_buffer(self, src, dst, src_offset=0, dst_offset=0, size=None,
                    wait_for=None, need_event=False):
        """Enqueues a command to copy from one buffer object to another.

        Parameters:
            src: source Buffer object.
            dst: destination Buffer object.
            src_offset: offset in bytes where to begin copying data from src.
            dst_offset: offset in bytes where to begin copying data into dst.
            size: number of bytes to copy.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        if size is None:    # assume smaller of either size
            # assert src.size == dst.size, 'buffer sizes must match'
            size = min(src.size - src_offset, dst.size - dst_offset)

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        n = self._lib.clEnqueueCopyBuffer(
                self.handle, src.handle, dst.handle, src_offset, dst_offset,
                size, n_events, wait_list, event)
        self.check_error(n, "clEnqueueCopyBuffer")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def copy_buffer_rect(self, src, dst, src_origin, dst_origin, region,
                         src_row_pitch=0, src_slice_pitch=0,
                         dst_row_pitch=0, dst_slice_pitch=0,
                         wait_for=None, need_event=False):
        """Enqueues a command to copy a 3D rectangular region from one
        buffer object to another.

        Parameters:
            src: source Buffer object.
            dst: destination Buffer object.
            src_origin: the (x in bytes, y, z) in the source buffer,
                        offset in bytes is computed as:
                        z * src_slice_pitch + y * src_row_pitch + x.
            dst_origin: the (x in bytes, y, z) in the destination buffer,
                        offset in bytes is computed as:
                        z * dst_slice_pitch + y * dst_row_pitch + x.
            region: the (width in bytes, height, depth)
                    of the rectangle being copied.
            src_row_pitch: the length of each source row in bytes,
                           if 0, region[0] will be used.
            src_slice_pitch: the length of each 2D source slice in bytes,
                             if 0, region[1] * src_row_pitch will be used.
            dst_row_pitch: the length of each destination row in bytes,
                           if 0, region[0] will be used.
            dst_slice_pitch: the length of each 2D destination slice in bytes,
                             if 0, region[1] * src_row_pitch will be used.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        _src_origin = cl.ffi.new("size_t[]", src_origin)
        _dst_origin = cl.ffi.new("size_t[]", dst_origin)
        _region = cl.ffi.new("size_t[]", region)
        n = self._lib.clEnqueueCopyBufferRect(
                self.handle, src.handle, dst.handle,
                _src_origin, _dst_origin, _region,
                src_row_pitch, src_slice_pitch,
                dst_row_pitch, dst_slice_pitch,
                n_events, wait_list, event)
        self.check_error(n, "clEnqueueCopyBufferRect")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def copy_image(self, src_image, dst_image,
                   src_origin=(0,0), dst_origin=(0,0), region=None,
                   wait_for=None, need_event=False):
        """Enqueues a command to copy an Image region into another Image.

        Parameters:
            src_image: Image object to copy from.
            dst_image: Image object to copy into.
            src_origin: (x,y,z) offset of image region copied from.
            src_origin: (x,y,z) offset of image region copied into.
            region: (x,y,z) size of copied image region.
            dst_offset: offset into buffer for copy.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False,
        """
        src_origin = tuple(src_origin) + (3-len(src_origin)) * (0,)
        dst_origin = tuple(dst_origin) + (3-len(dst_origin)) * (0,)

        if region is None:
            region = [ max(x, 1) for x in (
                src_image.width,
                src_image.height,
                src_image.depth) ]

        region = tuple(region) + (3-len(region)) * (1,)

        src_origin_struct = cl.ffi.new("size_t[3]", src_origin)
        dst_origin_struct = cl.ffi.new("size_t[3]", dst_origin)
        region_struct = cl.ffi.new("size_t[3]", region)

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        n = self._lib.clEnqueueCopyImage(
                self.handle, src_image.handle, dst_image.handle,
                src_origin_struct, dst_origin_struct, region_struct,
                n_events, wait_list, event)

        self.check_error(n, "clEnqueueCopyImage")

        return None if event == cl.ffi.NULL else Event(event[0])


    def copy_image_to_buffer(self, src_image, dst_buffer,
                   src_origin=(0,0), region=None, dst_offset=0,
                   wait_for=None, need_event=False):
        """Enqueues a command to copy an Image region into a Buffer.

        Parameters:
            src_image: Image object to copy from.
            dst_buffer: Buffer object to copy into.
            src_origin: (x,y,z) offset of copied image region.
            region: (x,y,z) size of copied image region.
            dst_offset: offset into buffer for copy.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False,
        """
        if region is None:
            region = [ max(x, 1) for x in (
                src_image.width,
                src_image.height,
                src_image.depth) ]

        region = tuple(region) + (3-len(region)) * (1,)
        src_origin = tuple(src_origin) + (3-len(src_origin)) * (0,)

        src_origin_struct = cl.ffi.new("size_t[3]", src_origin)
        region_struct = cl.ffi.new("size_t[3]", region)

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        n = self._lib.clEnqueueCopyImageToBuffer(
                self.handle, src_image.handle, dst_buffer.handle,
                src_origin_struct, region_struct, dst_offset,
                n_events, wait_list, event)

        self.check_error(n, "clEnqueueCopyImageToBuffer")

        return None if event == cl.ffi.NULL else Event(event[0])


    def copy_buffer_to_image(self, src_buffer, dst_image,
                   src_offset=0, dst_origin=(0,0), region=None,
                   wait_for=None, need_event=False):
        """Enqueues a command to copy an Image region into a Buffer.

        Parameters:
            src_buffer: Buffer object to copy from.
            dst_image:  Image object to copy to.
            src_offset: offset into buffer for copy.
            dst_origin: (x,y,z) offset of written image region.
            region:     (x,y,z) size of written image region.
            wait_for:   list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False,
        """
        if region is None:
            region = [ max(x, 1) for x in (
                dst_image.width,
                dst_image.height,
                dst_image.depth) ]

        region = tuple(region) + (3-len(region)) * (1,)
        dst_origin = tuple(dst_origin) + (3-len(dst_origin)) * (0,)

        dst_origin_struct = cl.ffi.new("size_t[3]", dst_origin)
        region_struct = cl.ffi.new("size_t[3]", region)

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        n = self._lib.clEnqueueCopyImageToBuffer(
                self.handle, src_buffer.handle, dst_image.handle,
                src_offset, dst_origin_struct, region_struct,
                n_events, wait_list, event)

        self.check_error(n, "clEnqueueCopyBufferToImage")

        return None if event == cl.ffi.NULL else Event(event[0])


    def fill_buffer(self, buf, pattern, pattern_size=None, size=None, offset=0, # FIXME: calculate size?
                    wait_for=None, need_event=False):
        """Enqueues a command to fill a region of a Buffer.

        Parameters:
            buf: Buffer object.
            pattern: a pointer to the data pattern of size pattern_size
                     in bytes, pattern will be used to fill a region in
                     buffer starting at offset and is size bytes in size
                     (numpy array or direct cffi pointer).
            pattern_size: pattern size in bytes.
            size: the size in bytes of region being filled in buf
                  and must be a multiple of pattern_size.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        if size is None:
            size = buf.size

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        pattern, pattern_size = CL.extract_ptr_and_size(pattern, pattern_size)
        n = self._lib.clEnqueueFillBuffer(
                self.handle, buf.handle, pattern, pattern_size, offset,
                size or (buf.size - offset), n_events, wait_list, event)
        self.check_error(n, "clEnqueueFillBuffer")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def svm_map(self, svm_ptr, flags, size=None, blocking=True,
                wait_for=None, need_event=False):
        """Enqueues a command that will allow the host to update a region
        of a SVM buffer.

        Parameters:
            svm_ptr: SVM object or numpy array or direct cffi pointer.
            flags: mapping flags.
            size: mapping size (may be None if svm_ptr is a numpy array).
            blocking: if the call would block until completion.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        if isinstance(svm_ptr, SVM):
            ptr = svm_ptr.handle
        else:
            ptr, size = CL.extract_ptr_and_size(svm_ptr, size)
        err = self._lib.clEnqueueSVMMap(
                self.handle, blocking, flags, ptr, size or svm_ptr.size,
                n_events, wait_list, event)
        self.check_error(err, "clEnqueueSVMMap")
        return None if event == cl.ffi.NULL else Event(event[0])


    def svm_unmap(self, svm_ptr, wait_for=None, need_event=False):
        """Unmaps previously mapped SVM buffer.

        Parameters:
            svm_ptr: pointer that was specified in a previous call to svm_map.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        if isinstance(svm_ptr, SVM):
            ptr = svm_ptr.handle
        else:
            ptr, _size = CL.extract_ptr_and_size(svm_ptr, 0)
        err = self._lib.clEnqueueSVMUnmap(
                self.handle, ptr, n_events, wait_list, event)
        self.check_error(err, "clEnqueueSVMUnmap")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def svm_memcpy(self, dst, src, size=None, blocking=True,
                   wait_for=None, need_event=False):
        """Enqueues a command to do a memcpy operation.

        Parameters:
            dst: destination (numpy array or direct cffi pointer).
            src: source (numpy array or direct cffi pointer).
            size: number of bytes to copy.
            blocking: if the call would block until completion.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        dst, sz_dst = CL.extract_ptr_and_size(dst, 0)
        src, sz_src = CL.extract_ptr_and_size(src, 0)
        size = size or min(sz_src, sz_dst)
        n = self._lib.clEnqueueSVMMemcpy(
                self.handle, blocking, dst, src, size, n_events, wait_list, event)
        self.check_error(n, "clEnqueueSVMMemcpy")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def svm_memfill(self, svm_ptr, pattern, pattern_size, size,     # TODO: remove need for explicit sizes
                    wait_for=None, need_event=False):
        """Enqueues a command to fill a region in memory with a pattern
        of a given pattern size.

        Parameters:
            svm_ptr: SVM object or numpy array or direct cffi pointer.
            pattern: a pointer to the data pattern of size pattern_size
                     in bytes (numpy array or direct cffi pointer).
            pattern_size: pattern size in bytes.
            size: the size in bytes of region being filled starting
                  with svm_ptr and must be a multiple of pattern_size.
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """
        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)
        if isinstance(svm_ptr, SVM):
            ptr = svm_ptr.handle
        else:
            ptr, _ = CL.extract_ptr_and_size(svm_ptr, 0)
        pattern, _ = CL.extract_ptr_and_size(pattern, 0)
        n = self._lib.clEnqueueSVMMemFill(
                self.handle, ptr, pattern, pattern_size, size,
                n_events, wait_list, event)
        self.check_error(n, "clEnqueueSVMMemFill")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def acquire_gl_objects(self, mem_objects, wait_for=None, need_event=False):
        """Acquire OpenCL memory objects that have been created from OpenGL objects.

        Parameters:
            mem_objects: iterable of MemObject instances to acquire control of
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        mem_object_arr = cl.ffi.new("cl_mem[]", len(mem_objects))
        for i, o in enumerate(mem_objects):
            mem_object_arr[i] = o.handle

        err = self._lib.clEnqueueAcquireGLObjects(
                    self.handle, len(mem_objects), mem_object_arr,
                    n_events, wait_list, event)
        self.check_error(err, "clEnqueueAcquireGLObjects")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def release_gl_objects(self, mem_objects, wait_for=None, need_event=False):
        """Release OpenCL memory objects that have been created from OpenGL objects.

        Parameters:
            mem_objects: iterable of MemObject instances to release control of
            wait_for: list of the Event objects to wait.
            need_event: return Event object or not.

        Returns:
            Event object or None if need_event == False.
        """

        event = cl.ffi.new("cl_event*") if need_event else cl.ffi.NULL
        wait_list, n_events = CL.get_wait_list(wait_for)

        mem_object_arr = cl.ffi.new("cl_mem[]", len(mem_objects))
        for i, o in enumerate(mem_objects):
            mem_object_arr[i] = o.handle

        err = self._lib.clEnqueueReleaseGLObjects(
                    self.handle, len(mem_objects), mem_object_arr,
                    n_events, wait_list, event)
        self.check_error(err, "clEnqueueReleaseGLObjects")
        return Event(event[0]) if event != cl.ffi.NULL else None


    def marker(self):
        """ Create an event dependent on all previous commands
            issued to this queue.
        """
        event = cl.ffi.new("cl_event*")
        n = self._lib.clEnqueueMarker(self.handle, event)
        self.check_error(n, "clEnqueueMarker")
        return event


    def barrier(self):
        """ Makes all following issued commands wait until
            previous commands in this queue complete
        """
        n = self._lib.clEnqueueBarrier(self.handle)
        self.check_error(n, "clEnqueueBarrier")


    def flush(self):
        """ Waits for all previous commands issued to this queue to
            be started on the device(s)
        """
        n = self._lib.clFlush(self.handle)
        self.check_error(n, "clFlush")


    def finish(self):
        """ Waits for all previous commands issued to this queue to complete.
        """
        n = self._lib.clFinish(self.handle)
        self.check_error(n, "clFinish")


    def _release(self):
        if self.handle is not None:
            self._lib.clReleaseCommandQueue(self.handle)
            self._handle = None


    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)


class MemObject(object):    # implements clGetMemObjectInfo interface
    @property
    def type(self):
        buf = cl.ffi.new("cl_mem_object_type *")
        self._get_mem_object_info(cl.CL_MEM_TYPE, buf)
        return buf[0]

    @property
    def flags(self):
        """
        Flags supplied for the creation of this buffer.
        """
        try:
            return self._flags
        except AttributeError:
            buf = cl.ffi.new("cl_mem_flags *")
            self._get_mem_object_info(cl.CL_MEM_FLAGS, buf)
            self._flags = int(buf[0])
            return self._flags

    @property
    def size(self):
        try:
            return self._size
        except AttributeError:
            buf = cl.ffi.new("size_t *")
            self._get_mem_object_info(cl.CL_MEM_SIZE, buf)
            self._size = int(buf[0])
            return self._size


    @property
    def host_ptr(self):
        buf = cl.ffi.new("void* *")
        self._get_mem_object_info(cl.CL_MEM_HOST_PTR, buf)
        return buf[0]


    @property
    def map_count(self):
        buf = cl.ffi.new("cl_uint *")
        self._get_mem_object_info(cl.CL_MEM_MAP_COUNT, buf)
        return int(buf[0])


    @property
    def reference_count(self):
        buf = cl.ffi.new("cl_uint *")
        self._get_mem_object_info(cl.CL_MEM_REFERENCE_COUNT, buf)
        return int(buf[0])


    @property
    def context(self):
        try:
            return self._context
        except ValueError:
            buf = cl.ffi.new("cl_context *")
            self._get_mem_object_info(cl.CL_MEM_CONTEXT, buf)
            self._context = Context.from_cl_context(buf[0])
            return self._context


    def _get_mem_object_info(self, code, buf):
       sz = cl.ffi.new("size_t *")
       err = self._lib.clGetMemObjectInfo( self.handle,
           code, cl.ffi.sizeof(buf), buf, sz)
       self.check_error(err, 'clGetMemObjectInfo')
       return sz[0]


    @property
    def gl_object_info(self):
        assert self.from_gl, 'Object is not shared with OpenGL'
        gl_object_type = cl.ffi.new('cl_gl_object_type *')
        gl_object_name = cl.ffi.new('cl_GLuint *')

        err = self._lib.clGetGLObjectInfo( self.handle, gl_object_type, gl_object_name )
        self.check_error(err, 'clGetGLObjectInfo')
        return gl_object_type[0], gl_object_name[0]


class Buffer(CL, MemObject):
    """Holds OpenCL buffer.

    Attributes:
        context: Context object associated with this buffer.
        flags: flags supplied for the creation of this buffer.
        host_array: host array reference, such as numpy array,
                    will be stored only if flags include CL_MEM_USE_HOST_PTR.
        size: size of the host array.
        parent: parent buffer if this one should be created as sub buffer.
        origin: origin of the sub buffer if parent is not None.
        _n_refs: reference count as a workaround for possible
                 incorrect destructor call order, see
                 http://bugs.python.org/issue23720
                 (weakrefs do not help here).
    """

    def _init_empty(self, context, flags, parent=None):  # __init__ boilerplate
        super(Buffer, self).__init__()
        context._add_ref(self)
        self._n_refs = 1
        self._parent = parent
        if parent is not None:
            parent._add_ref(self)
        self._context = context
        self._flags = flags
        return self


    def __init__(self, context, flags, host_array=None, size=None,
                 parent=None, origin=0):
        self._init_empty(context, flags, parent)
        self._host_array = (host_array if flags & cl.CL_MEM_USE_HOST_PTR
                            else None)
        host_ptr, size = CL.extract_ptr_and_size(host_array, size)
        self._size = size
        self._origin = origin
        err = cl.ffi.new("cl_int *")

        if parent is None:
            self._handle = self._lib.clCreateBuffer(
                            context.handle, flags, size, host_ptr, err)
        else:
            info = cl.ffi.new("size_t[]", 2)
            info[0] = origin
            info[1] = size
            self._handle = self._lib.clCreateSubBuffer(
                            parent.handle, flags, cl.CL_BUFFER_CREATE_TYPE_REGION,
                            info, err)

        self.check_error(err[0], 'clCreateBuffer', release=True)


    @classmethod
    def from_gl_buffer(cls, context, flags, bufferobj):
        """Creates a Buffer from an OpenGL buffer object.

        Parameters:
            flags: memory access descriptor flags
            renderbuffer: The name of a GL renderbuffer object.
                        The renderbuffer storage must be specified before the
                        image object can be created. The renderbuffer format and
                        dimensions will be used to create the 2D image object.
        Returns:
            Buffer object.
        """
        self = cls.__new__(cls)             # manually construct empty object
        self._init_empty(context, flags)
        self._gl_buffer = cl.ffi.cast('cl_GLuint', bufferobj)  # host_glbuffer?

        err = cl.ffi.new("cl_int *")
        self._handle = self._lib.clCreateFromGLBuffer(
                            context.handle, flags, self._gl_buffer, err)

        self.check_error(err[0], 'clCreateFromGLBuffer', release=True)
        return self


    def create_sub_buffer(self, origin=0, size=None, flags=0):
        """Creates subbufer from the region of the original buffer.

        Parameters:
            flags: flags for the creation of this buffer
                   (0 - inherit all from the original buffer).
            origin: offset in bytes in the original buffer
            size: size in bytes of the new buffer.
        """
        return Buffer(  self._context, flags,
                        self._host_array, size or self.size,
                        self, origin)


    def _add_ref(self, obj):
        self._n_refs += 1


    def _del_ref(self, obj):
        with cl.lock:
            self._n_refs -= 1
            n_refs = self._n_refs
        if n_refs <= 0:
            self._release()


    @property
    def host_array(self):
        """
        Host array reference, such as numpy array,
        will be stored only if flags include CL_MEM_USE_HOST_PTR.
        """
        return self._host_array


    @property
    def host_array_size(self):
        """
        Size of the host array.
        """
        return self._size

    @property
    def parent(self):
        """Returns parent buffer if this buffer is a sub buffer.
        """
        return self._parent

    def _release(self):
        if self.handle is not None:
            if self.parent is not None and self.parent.handle is None:
                raise SystemError("Incorrect destructor call order detected")
            self._lib.clReleaseMemObject(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._del_ref(self)
        if self.parent is not None:
            self.parent._del_ref(self)
        self.context._del_ref(self)


class skip(object):
    """A marker to skip setting arguments in Kernel.set_args.
    Passing in the class type makes set_args to skip setting one argument;
    passing skip(n) makes set_args skip n arguments.
    """
    def __init__(self, number):
        self.number = number

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, value):
        if value < 1:
            raise ValueError("number must be greater than 0")
        self._number = value


class Image(CL, MemObject):
    """Holds OpenCL image.

    Attributes:
        context: Context object associated with this image.
        flags: flags supplied for the creation of this image.
        image_format: A 2-tuple of integers.
                    (image_channel_order, image_channel_data_type)
        image_desc: A 10-tuple containing the values of a cl_image_desc struct.
        host_array: host array reference, such as numpy array,
                    will be stored only if flags include CL_MEM_USE_HOST_PTR.
        _n_refs: reference count as a workaround for possible
                 incorrect destructor call order, see
                 http://bugs.python.org/issue23720
                 (weakrefs do not help here).
    """

    IMGRW_SUFFIX_MAP = {
		cl.CL_SNORM_INT8: 'f', cl.CL_SNORM_INT16: 'f', cl.CL_UNORM_INT8: 'f', cl.CL_UNORM_INT16: 'f',
		cl.CL_UNORM_INT24:'f', cl.CL_UNORM_SHORT_565: 'f', cl.CL_UNORM_SHORT_555: 'f',
		cl.CL_UNORM_INT_101010:'f', cl.CL_UNORM_INT_101010_2: 'f', cl.CL_FLOAT: 'f',
		cl.CL_SIGNED_INT8: 'i', cl.CL_SIGNED_INT16: 'i', cl.CL_SIGNED_INT32: 'i',
		cl.CL_UNSIGNED_INT8: 'ui', cl.CL_UNSIGNED_INT16: 'ui', cl.CL_UNSIGNED_INT32: 'ui',
        cl.CL_HALF_FLOAT: 'h'
	}

    FORMAT_DTYPE_MAP = {
		cl.CL_UNORM_SHORT_565:  '>u2',  cl.CL_UNORM_SHORT_555:    '>u2',
		cl.CL_UNORM_INT_101010: '>u4',  cl.CL_UNORM_INT_101010_2: '>u4',         # FIXME: are packed formats big-endian?
		cl.CL_SNORM_INT8:        'i2',  cl.CL_SNORM_INT16:         'i2',
        cl.CL_UNORM_INT8:        'u1',  cl.CL_UNORM_INT16:         'u2',
        cl.CL_HALF_FLOAT:        'f2',  cl.CL_FLOAT:               'f4',
		cl.CL_SIGNED_INT8:   'i1', cl.CL_SIGNED_INT16:   'i2', cl.CL_SIGNED_INT32:   'i4',
		cl.CL_UNSIGNED_INT8: 'u1', cl.CL_UNSIGNED_INT16: 'u2', cl.CL_UNSIGNED_INT32: 'u4'
	}

    ORDER_SIZE_MAP = {  # CL_Rx, CL_RGx, CL_RGBx, CL_sRGBx are the same but have alpha=0 at their borders (?)
       cl.CL_DEPTH:     1,  cl.CL_DEPTH_STENCIL: 1,
       cl.CL_INTENSITY: 1,  cl.CL_LUMINANCE:     1,
       cl.CL_R:         1,  cl.CL_Rx:    1, cl.CL_A:     1,
       cl.CL_RG:        2,  cl.CL_RGx:   2, cl.CL_RA:    2,
       cl.CL_RGB:       3,  cl.CL_RGBx:  3, cl.CL_sRGB:  3, cl.CL_sRGBx: 3,
       cl.CL_RGBA:      4,  cl.CL_sRGBA: 4, cl.CL_BGRA:  4, cl.CL_sBGRA: 4,
       cl.CL_ABGR:      4,  cl.CL_ARGB:  4
    }

    IMG_CL_TYPE = { 'f': 'float', 'h': 'half', 'i':'int', 'ui':'uint' }

    @property
    def _cl_type_suffix(self):
        return self.IMGRW_SUFFIX_MAP[self.format.image_channel_data_type]

    @property
    def cl_image_read_func_name(self):
        return 'image_read' + self._cl_type_suffix

    @property
    def cl_image_write_func_name(self):
        return 'image_write' + self._cl_type_suffix

    @property
    def cl_type_name(self):
        return self.IMG_CL_TYPE[self._cl_type_suffix]

    @property
    def dtype(self):
        imfmt = self.format
        dtype_str = self.FORMAT_DTYPE_MAP[ imfmt.image_channel_data_type ]
        order_size =  self.ORDER_SIZE_MAP[ imfmt.image_channel_order ]
        if order_size == 1 or '>' in dtype_str:
            return np.dtype(dtype_str)
        else:
            return np.dtype((dtype_str, order_size))

    @property
    def shape(self):
        raw_shape = (self.depth, self.height, self.width)
        return tuple( s for s in raw_shape if s > 1 )


    def create_np_array(self):
        return np.zeros(self.shape, self.dtype)


    def _init_empty(self, context, flags):  # __init__ boilerplate
        super(Image, self).__init__()
        context._add_ref(self)
        self._n_refs = 1
        self._context = context
        self._flags = flags
        return self


    def __init__(self, context, flags, image_format, image_desc, host_array=None):
        self._init_empty(context, flags)
        self._host_array = (host_array if flags & cl.CL_MEM_USE_HOST_PTR != 0 else None)
        host_ptr = CL.extract_ptr(host_array)

        self.image_format = ensure_type("cl_image_format *", image_format)
        self.image_desc = ensure_type("cl_image_desc *", image_desc)   # will the cast of buffer/mem_object work?

        err = cl.ffi.new("cl_int *")
        self._handle = self._lib.clCreateImage(
                            context.handle, flags,
                            self.image_format, self.image_desc, host_ptr, err)
        self.check_error(err[0], 'clCreateImage', release=True)

        self.from_gl = False


    @classmethod
    def from_gl_renderbuffer(cls, context, flags, renderbuffer):
        """Creates a 2D Image from an OpenGL renderbuffer object.

        Parameters:
            flags: memory access descriptor flags
            renderbuffer: The name of a GL renderbuffer object.
                        The renderbuffer storage must be specified before the
                        image object can be created. The renderbuffer format and
                        dimensions will be used to create the 2D image object.
        Returns:
            Image object.
        """
        self = cls.__new__(cls)             # manually construct empty object
        self._init_empty(context, flags)
        self._host_renderbuffer = renderbuffer   # host_glbuffer?

        err = cl.ffi.new("cl_int *")

        self._handle = self._lib.clCreateFromGLRenderbuffer(
                            context.handle, flags, renderbuffer, err)

        self.check_error(err[0], 'clCreateFromGLRenderbuffer', release=True)
        self.from_gl = True
        return self


    @classmethod
    def from_gl_texture(cls, context, flags, texture_target, miplevel, texture):
        """Creates a 2D Image from an OpenGL texture object.

        Parameters:
            flags: memory access descriptor flags.
            texture_target: image type of texture.
            miplevel: mipmap level to be used. Usually 0.
            texture: The name of a GL texture object.
                        Can be 1D, 2D, 3D, 1D array, 2D array, cubemap, rectangle
                        or buffer texture object. Must be a complete texture.
        Returns:
            Image object.
        """
        self = cls.__new__(cls)             # manually construct empty object
        self._init_empty(context, flags)
        self._host_texture = texture        # host_glbuffer?

        err = cl.ffi.new("cl_int *")

        self._handle = self._lib.clCreateFromGLTexture(
                            context.handle, flags, texture_target,
                            miplevel, texture, err)

        self.check_error(err[0], 'clCreateFromGLTexture', release=True)
        self.from_gl = True
        return self


    def _add_ref(self, obj):
        self._n_refs += 1

    def _del_ref(self, obj):
        with cl.lock:
            self._n_refs -= 1
            n_refs = self._n_refs
        if n_refs <= 0:
            self._release()

    @property
    def context(self):
        """
        Context object associated with this buffer.
        """
        return self._context


    @property
    def host_array(self):
        """
        Host array reference, such as numpy array,
        will be stored only if flags include CL_MEM_USE_HOST_PTR.
        """
        return self._host_array


    @property
    def format(self):
        """Returns image format descriptor specified when image is created.
        """
        buf = cl.ffi.new("cl_image_format *")
        self._get_image_info(cl.CL_IMAGE_FORMAT, buf)
        return buf[0]


    @property
    def element_size(self):
        "Return size of each element of the image memory object."
        buf = cl.ffi.new("size_t *")
        self._get_image_info(cl.CL_IMAGE_ELEMENT_SIZE, buf)
        return int(buf[0])

    @property
    def row_pitch(self):
        "Return size in bytes of a row of elements of the image."
        buf = cl.ffi.new("size_t *")
        self._get_image_info(cl.CL_IMAGE_ROW_PITCH, buf)
        return int(buf[0])

    @property
    def slice_pitch(self):
        """Return size in bytes of a 2D slice for a 3D image.
           For a 2D image this will be 0."""
        buf = cl.ffi.new("size_t *")
        self._get_image_info(cl.CL_IMAGE_SLICE_PITCH, buf)
        return int(buf[0])

    @property
    def width(self):
        "Return width of image in pixels."
        buf = cl.ffi.new("size_t *")
        self._get_image_info(cl.CL_IMAGE_WIDTH, buf)
        return int(buf[0])

    @property
    def height(self):
        "Return height of image in pixels."
        buf = cl.ffi.new("size_t *")
        self._get_image_info(cl.CL_IMAGE_HEIGHT, buf)
        return int(buf[0])

    @property
    def depth(self):
        """Return depth of the image in pixels.
           For a 2D image, depth equals 0."""
        buf = cl.ffi.new("size_t *")
        self._get_image_info(cl.CL_IMAGE_DEPTH, buf)
        return int(buf[0])

    @property
    def d3d10_subresource(self):
        "Return size of each element of the image memory object."
        buf = cl.ffi.new("ID3D10Resource *")
        self._get_image_info(cl.CL_IMAGE_D3D10_SUBRESOURCE_KHR, buf)
        return int(buf[0])  # should i do this?

    def _get_image_info(self, code, buf):
        sz = cl.ffi.new("size_t *")
        err = self._lib.clGetImageInfo( self.handle,
            code, cl.ffi.sizeof(buf), buf, sz)
        self.check_error(err, 'clGetImageInfo')
        return sz[0]

    def _release(self):
        if self.handle is not None:
            self._lib.clReleaseMemObject(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._del_ref(self)
        self.context._del_ref(self)


class WorkGroupInfo(CL):
    """Some information about the kernel concerning the specified device.
    """
    def __init__(self, kernel, device):
        super(WorkGroupInfo, self).__init__()
        self._kernel = kernel
        self._device = device

    @property
    def kernel(self):
        return self._kernel

    @property
    def device(self):
        return self._device

    @property
    def global_work_size(self):
        """Returns the maximum global size that can be used to execute a kernel
           on this device.

        Raises:
            CLRuntimeError: when device is not a custom device or
                            kernel is not a built-in kernel.
        """
        buf = cl.ffi.new("size_t[]", 3)
        self._get_workgroup_info(cl.CL_KERNEL_GLOBAL_WORK_SIZE, buf)
        return int(buf[0]), int(buf[1]), int(buf[2])

    @property
    def work_group_size(self):
        """Returns the maximum global size that can be used to execute a kernel
           on this device.
        """
        buf = cl.ffi.new("size_t *")
        self._get_workgroup_info(cl.CL_KERNEL_WORK_GROUP_SIZE, buf)
        return int(buf[0])

    @property
    def compile_work_group_size(self):
        """Returns the work-group size specified by the
           __attribute__((reqd_work_group_size(X, Y, Z))) qualifier.
        """
        buf = cl.ffi.new("size_t[]", 3)
        self._get_workgroup_info(cl.CL_KERNEL_COMPILE_WORK_GROUP_SIZE, buf)
        return int(buf[0]), int(buf[1]), int(buf[2])

    @property
    def local_mem_size(self):
        """Returns the amount of local memory in bytes being used by a kernel.
        """
        buf = cl.ffi.new("uint64_t *")
        self._get_workgroup_info(cl.CL_KERNEL_LOCAL_MEM_SIZE, buf)
        return int(buf[0])

    @property
    def preferred_work_group_size_multiple(self):
        """Returns the preferred multiple of workgroup size for launch.
        """
        buf = cl.ffi.new("size_t *")
        self._get_workgroup_info(cl.CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, buf)
        return int(buf[0])

    @property
    def private_mem_size(self):
        """Returns the minimum amount of private memory, in bytes,
           used by each workitem in the kernel.
        """
        buf = cl.ffi.new("uint64_t *")
        self._get_workgroup_info(cl.CL_KERNEL_PRIVATE_MEM_SIZE, buf)
        return int(buf[0])

    def _get_workgroup_info(self, code, buf):
        sz = cl.ffi.new("size_t *")
        err = self._lib.clGetKernelWorkGroupInfo(
                    self.kernel.handle, self.device.handle, code,
                    cl.ffi.sizeof(buf), buf, sz)
        self.check_error(err, "clGetKernelWorkGroupInfo")
        return sz[0]


class Kernel(CL):
    """Holds OpenCL kernel.

    Attributes:
        program: Program object associated with this kernel.
        name: kernel name in the program.
    """

    def __init__(self, program, name):
        super(Kernel, self).__init__()
        self._program = program
        self._name = name
        err = cl.ffi.new("cl_int *")
        ss = cl.ffi.new("char[]", name.encode("utf-8"))
        self._handle = self._lib.clCreateKernel(program.handle, ss, err)
        self.check_error(err[0], "clCreateKernel", release=True)
#            self._handle = None

    @property
    def program(self):
        """
        Program object associated with this kernel.
        """
        return self._program

    @property
    def name(self):
        """
        kernel name in the program.
        """
        return self._name

    @property
    def reference_count(self):
        buf = cl.ffi.new("cl_uint *")
        self._get_kernel_info(cl.CL_KERNEL_REFERENCE_COUNT, buf)
        return buf[0]

    @property
    def num_args(self):
        buf = cl.ffi.new("size_t *")
        self._get_kernel_info(cl.CL_KERNEL_NUM_ARGS, buf)
        return buf[0]

    @property
    def attributes(self):
        buf = cl.ffi.new("char[]", 4096)
        self._get_kernel_info(cl.CL_KERNEL_ATTRIBUTES, buf)
        return cl.ffi.string(buf).decode("utf-8", "replace").strip()

    def get_work_group_info(self, device):
        return WorkGroupInfo(self, device)

    def set_arg(self, idx, vle, size=None):
        """Sets kernel argument.

        Parameters:
            idx: index of the kernel argument (zero-based).
            vle: kernel argument:
                 - for buffers should be an instance of Buffer,
                 - for scalars should be a numpy array slice
                   (k[0:1] for example),
                 - for NULL should be None,
                 - may be cffi pointer also, in such case size should be set.
            size: size of the vle (may be None for buffers and scalars).
        """
        if isinstance(vle, Buffer) or isinstance(vle, Image) or isinstance(vle, Pipe):
            arg_value = cl.ffi.new("cl_mem*", vle.handle)
            arg_size = cl.ffi.sizeof("cl_mem")
        elif hasattr(vle, "__array_interface__"):   # constant data
            arg_value = cl.ffi.cast("const void*",
                                    vle.__array_interface__["data"][0])
            arg_size = vle.nbytes if size is None else size
        elif vle is None:
            arg_value = cl.ffi.NULL
            arg_size = cl.ffi.sizeof("cl_mem") if size is None else size
        elif isinstance(vle, cl.ffi.CData):  # cffi object
            arg_value = cl.ffi.cast("const void*", vle)
            if size is None:
                raise ValueError("size should be set in case of cffi pointer")
            arg_size = size
        elif isinstance(vle, SVM):
            return self.set_arg_svm(idx, vle)
        elif isinstance(vle, Sampler):
            arg_value = cl.ffi.new("const void *", vle.sampler)
            arg_size = cl.ffi.sizeof("sampler_t")
        else:
            raise ValueError("vle should be of type Buffer, Image, Pipe, SVM, "
                             "numpy array, cffi pointer or None "
                             "in Kernel::set_arg()")
        n = self._lib.clSetKernelArg(self.handle, idx, arg_size, arg_value)
        self.check_error(n, "clSetKernelArg", " for argument %d" % idx)

    def set_arg_svm(self, idx, svm_ptr):
        """Sets SVM pointer as the kernel argument.

        Parameters:
            idx: index of the kernel argument (zero-based).
            svm_ptr: SVM object or numpy array or direct cffi pointer.
        """
        if isinstance(svm_ptr, SVM):
            ptr = svm_ptr.handle
        else:
            ptr, _size = CL.extract_ptr_and_size(svm_ptr, 0)
        err = self._lib.clSetKernelArgSVMPointer(self.handle, idx, ptr)
        if err:
            raise CLRuntimeError(
                "clSetKernelArgSVMPointer(%d, %s) failed with error %s" %
                (idx, repr(svm_ptr), CL.get_error_description(err)), err)

    def set_args(self, *args):
        i = 0
        for arg in args:
            if arg is skip:
                i += 1
                continue
            if isinstance(arg, skip):
                i += arg.number
                continue
            if isinstance(arg, tuple) and len(arg) == 2:
                self.set_arg(i, *arg)
            else:
                self.set_arg(i, arg)
            i += 1

    def _release(self):
        if self.handle is not None:
            self._lib.clReleaseKernel(self.handle)
            self._handle = None

    def _get_kernel_info(self, code, buf):
        sz = cl.ffi.new("size_t *")
        err = self._lib.clGetKernelInfo(
                    self.handle, code, cl.ffi.sizeof(buf), buf, sz)
        self.check_error(err, "clGetKernelInfo")
        return sz[0]

    def __del__(self):
        self._release()

    def __repr__(self):
        return '<opencl4py.Kernel %r>' % self.name


class Program(CL):
    """Holds OpenCL program.

    Attributes:
        context: Context object associated with this program.
        devices: list of Device objects associated with this program.
        build_logs: list of program build logs (same length as devices list).
        src: program source.
        include_dirs: list of include dirs.
        options: additional build options.
        binary: False if the program should be created from source; otherwise,
                src is interpreted as precompiled binaries iterable.
    """

    def __init__(self, context, devices, src, include_dirs=(), options="",
                 binary=False):
        super(Program, self).__init__()
        context._add_ref(self)
        self._context = context
        self._devices = devices
        self._src = src.encode("utf-8") if not binary else None
        self._include_dirs = list(include_dirs)
        self._options = options.strip().encode("utf-8")
        self._build_logs = []
        if not binary:
            self._create_program_from_source()
        else:
            self._create_program_from_binary(src)

    @property
    def context(self):
        """
        Context object associated with this program.
        """
        return self._context

    @property
    def devices(self):
        """
        List of Device objects associated with this program.
        """
        return self._devices

    @property
    def build_logs(self):
        """
        List of program build logs (same length as devices list).
        """
        return self._build_logs

    @property
    def source(self):
        """
        Program source.
        """
        return self._src

    @property
    def include_dirs(self):
        """
        List of include dirs.
        """
        return self._include_dirs

    @property
    def options(self):
        """
        Additional build options.
        """
        return self._options

    @property
    def reference_count(self):
        buf = cl.ffi.new("cl_uint *")
        self._get_program_info(cl.CL_PROGRAM_REFERENCE_COUNT, buf)
        return buf[0]

    @property
    def num_kernels(self):
        buf = cl.ffi.new("size_t *")
        self._get_program_info(cl.CL_PROGRAM_NUM_KERNELS, buf)
        return buf[0]

    @property
    def kernel_names(self):
        buf = cl.ffi.new("char[]", 4096)
        self._get_program_info(cl.CL_PROGRAM_KERNEL_NAMES, buf)
        names = cl.ffi.string(buf).decode("utf-8", "replace")
        return names.split(';')

    @property
    def binaries(self):
        sizes = cl.ffi.new("size_t[]", len(self.devices))
        self._get_program_info(cl.CL_PROGRAM_BINARY_SIZES, sizes)
        buf = cl.ffi.new("char *[]", len(self.devices))
        bufr = []  # to hold the references to cffi arrays
        for i in range(len(self.devices)):
            bufr.append(cl.ffi.new("char[]", sizes[i]))
            buf[i] = bufr[-1]
        self._get_program_info(cl.CL_PROGRAM_BINARIES, buf)
        bins = []
        for i in range(len(self.devices)):
            bins.append(bytes(cl.ffi.buffer(buf[i], sizes[i])[0:sizes[i]]))
        del bufr
        return bins

    def get_kernel(self, name):
        """Returns Kernel object from its name.
        """
        return Kernel(self, name)

    def _get_program_info(self, code, buf):
        sz = cl.ffi.new("size_t *")
        err = self._lib.clGetProgramInfo(self.handle, code,
                                         cl.ffi.sizeof(buf), buf, sz)
        self.check_error(err, "clGetProgramInfo")
        return sz[0]

    def _get_build_logs(self, device_list):
        del self.build_logs[:]
        log = cl.ffi.new("char[]", 65536)
        sz = cl.ffi.new("size_t *")
        for dev in device_list:
            e = self._lib.clGetProgramBuildInfo(
                    self.handle, dev, cl.CL_PROGRAM_BUILD_LOG,
                    cl.ffi.sizeof(log), log, sz)
            if e or sz[0] <= 0:
                self.build_logs.append("")
                continue
            self.build_logs.append(cl.ffi.string(log).decode("utf-8",
                                                             "replace"))

    def _create_program_from_source(self):
        err = cl.ffi.new("cl_int *")
        srcptr = cl.ffi.new("char[]", self.source)
        strings = cl.ffi.new("char*[]", 1)
        strings[0] = srcptr
        self._handle = self._lib.clCreateProgramWithSource(
                            self.context.handle, 1, strings, cl.ffi.NULL, err)
        del srcptr
        self.check_error(err[0], "clCreateProgramWithSource", release=True)

        options = self.options.decode("utf-8")
        for dirnme in self.include_dirs:
            if not len(dirnme):
                continue
            options += " -I " + (dirnme if dirnme.find(" ") < 0
                                 else "\'%s\'" % dirnme)
        options = options.encode("utf-8")
        n_devices = len(self.devices)
        device_list = cl.ffi.new("cl_device_id[]", n_devices)
        for i, dev in enumerate(self.devices):
            device_list[i] = dev.handle
        err = self._lib.clBuildProgram(self.handle, n_devices, device_list,
                                       options, cl.ffi.NULL, cl.ffi.NULL)
        del options
        self._get_build_logs(device_list)

        logstr = "\nLogs are:\n%s\nSource was:\n%s\n" % (
                    "\n".join(self.build_logs),
                    self.source.decode("utf-8") )

        self.check_error(err, 'clBuildProgram', logstr, release=True)


    def _create_program_from_binary(self, src):
        count = len(self.devices)
        if count != len(src):
            raise ValueError("You have supplied %d binaries for %d devices" %
                             (len(src), count))
        device_list = cl.ffi.new("cl_device_id[]", count)
        for i, dev in enumerate(self.devices):
            device_list[i] = dev.handle
        lengths = cl.ffi.new("size_t[]", count)
        for i, b in enumerate(src):
            lengths[i] = len(b)
        binaries_ffi = cl.ffi.new("unsigned char *[]", count)
        # The following 4 lines are here to prevent Python
        # from garbage collecting binaries_ffi[:]
        binaries_ref = []
        for i, b in enumerate(src):
            binaries_ref.append(cl.ffi.new("unsigned char[]", b))
            binaries_ffi[i] = binaries_ref[-1]
        binary_status = cl.ffi.new("cl_int[]", count)
        err = cl.ffi.new("cl_int *")
        self._handle = self._lib.clCreateProgramWithBinary(
                            self.context.handle, count, device_list, lengths,
                            binaries_ffi, binary_status, err)
        if err[0]:
            self._handle = None
            statuses = [CL.get_error_name_from_code(s) for s in binary_status]
            raise CLRuntimeError("clCreateProgramWithBinary() failed with "
                                 "error %s; status %s" % (
                                     CL.get_error_description(err[0]),
                                     ", ".join(statuses)),
                                 err[0])
        err = self._lib.clBuildProgram(self.handle, count, device_list,
                                       self.options, cl.ffi.NULL, cl.ffi.NULL)
        del binaries_ref
        self._get_build_logs(device_list)
        self.check_error(err, "clBuildProgram", release=True)

    def _release(self):
        if self.handle is not None:
            self._lib.clReleaseProgram(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)


class Pipe(CL):
    """Holds OpenCL pipe.

    Attributes:
        context: Context object associated with this pipe.
        flags: flags for a pipe;
               as of OpenCL 2.0 only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
               CL_MEM_READ_WRITE, and CL_MEM_HOST_NO_ACCESS can be specified
               when creating a pipe object (0 defaults to CL_MEM_READ_WRITE).
        packet_size: size in bytes of a pipe packet (must be greater than 0).
        max_packets: maximum number of packets the pipe can hold
                     (must be greater than 0).
    """
    def __init__(self, context, flags, packet_size, max_packets):
        super(Pipe, self).__init__()
        context._add_ref(self)
        self._context = context
        self._flags = flags
        self._packet_size = packet_size
        self._max_packets = max_packets
        err = cl.ffi.new("cl_int *")
        self._handle = self._lib.clCreatePipe(
                            context.handle, flags, packet_size, max_packets,
                            cl.ffi.NULL, err)
        self.check_error(err[0], "clCreatePipe", release=True)

    @property
    def context(self):
        return self._context

    @property
    def flags(self):
        return self._flags

    @property
    def packet_size(self):
        return self._packet_size

    @property
    def max_packets(self):
        return self._max_packets

    def _release(self):
        if self.handle is not None:
            self._lib.clReleaseMemObject(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)


class SVM(CL):
    """Holds shared virtual memory (SVM) buffer.

    Attributes:
        handle: pointer to the created buffer.
        context: Context object associated with this buffer.
        flags: flags for a buffer.
        size: size in bytes of the SVM buffer to be allocated.
        alignment: the minimum alignment in bytes (can be 0).
    """
    def __init__(self, context, flags, size, alignment=0):
        super(SVM, self).__init__()
        context._add_ref(self)
        self._context = context
        self._flags = flags
        self._size = size
        self._alignment = alignment
        self._handle = self._lib.clSVMAlloc(
                            context.handle, flags, size, alignment)
        if self._handle == cl.ffi.NULL:
            self._handle = None
            raise CLRuntimeError("clSVMAlloc() failed", cl.CL_INVALID_VALUE)

    @property
    def context(self):
        return self._context

    @property
    def flags(self):
        return self._flags

    @property
    def size(self):
        return self._size

    @property
    def alignment(self):
        return self._alignment

    @property
    def buffer(self):
        """Returns buffer object from this SVM pointer.

        You can supply it to numpy.frombuffer() for example,
        but be sure that destructor of an SVM object is called
        after the last access to that numpy array.
        """
        return cl.ffi.buffer(self.handle, self.size)

    def _release(self):
        if self.handle is not None and self.context.handle is not None:
            self._lib.clSVMFree(self.context.handle, self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)




class Context(CL):
    """Holds OpenCL context.

    Attributes:
        platform: Platform object associated with this context.
        devices: list of Device object associated with this context.
        _n_refs: reference count as a workaround for possible
                 incorrect destructor call order, see
                 http://bugs.python.org/issue23720
                 (weakrefs do not help here).
    """
    @staticmethod
    def _properties_list(*args):
        args = tuple(args) + (0,)   # list terminator
        pobj = cl.ffi.new("cl_context_properties[]",len(args))
        for i, a in enumerate(args):
            pobj[i] = cl.ffi.cast("cl_context_properties", a)
        return pobj

    def _init_empty(self):
        super(Context, self).__init__()
        self._n_refs = 1

    def _get_context_info(self, code, buf):
        sz = cl.ffi.new("size_t *")
        getbytes = cl.ffi.sizeof(buf) if buf else 0  # TODO: change others to use this
        err = self._lib.clGetContextInfo(self.handle, code,
                                         getbytes, buf, sz)
        self.check_error(err, "clGetContextInfo")
        return sz[0]

    @property
    def devices(self):
        """
        List of Device object associated with this context.
        """
        try:
            return self._devices
        except AttributeError:
            sz_ids = self._get_context_info( cl.CL_CONTEXT_DEVICES, cl.ffi.NULL )
            ids = cl.ffi.new( "cl_device_id[]", sz_ids )
            sz_ret = self._get_context_info( cl.CL_CONTEXT_DEVICES, ids )
            assert sz_ret == sz_ids
            self._devices = [ Device(i) for i in ids ]
            return self._devices

    @property
    def platform(self):
        """
        Platform object associated with this context.
        """
        try:
            return self._platform
        except AttributeError:
            self._platform = self.devices[0].platform
            assert all([ dev.platform == self._platform for dev in self.devices ])
            return self._platform

    @classmethod
    def from_cl_context(cls, context_id):
        self = cls.__new__(cls)
        self._init_empty()
        self._handle = context_id
        assert self.devices         # populate device list
        assert self.platform        # populate platform


    @classmethod
    def from_current_gl_context(cls, platform=None, egl=False):       #TODO: add multi-device support
        if platform is None:
            for platform in Platforms.get_ids():
                try:
                    return Context.from_current_gl_context(platform)
                except RuntimeError:
                    pass
            else:
                raise RuntimeError( "Could not find any platform able to use current GL context" )

        if isinstance(platform, Platform):
            cl_platform = platform.handle
        else:
            cl_platform = platform
            platform = Platform(cl_platform)

        if 'cl_khr_gl_sharing' not in platform.extensions:  # platform can return
            raise RuntimeError('Platform "%s" does not support the "cl_khr_gl_sharing" extension', platform.name)

        self = cls.__new__(cls)
        self._init_empty()
        err = cl.ffi.new("cl_int *")

        platform_props = (cl.CL_CONTEXT_PLATFORM, cl_platform) if cl_platform else ()

        if sys.platform == 'darwin':
            gl_context = self._gllib.CGLGetCurrentContext()
            gl_sharegroup = self._gllib.CGLGetShareGroup( gl_context )
            plist = Context._properties_list(
                cl.CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, gl_sharegroup,
                *platform_props )

            self._handle = self._lib.clCreateContext(
                plist, 0, cl.ffi.NULL, cl.ffi.NULL, cl.ffi.NULL, err)
            self.check_error(err[0], "clCreateContext", release=True)

            sizeof_device_id = cl.ffi.sizeof("cl_device_id")
            cl_device_id = cl.ffi.new("cl_device_id *")
            size_ret = cl.ffi.new("size_t *")

            status = self._lib.clGetGLContextInfoAPPLE(
                self._handle, gl_context,
                cl.CL_CGL_DEVICE_FOR_CURRENT_VIRTUAL_SCREEN_APPLE,
                sizeof_device_id, cl_device_id, size_ret )
            self.check_error(status, "clGetGLContextInfoAPPLE", release=True)
            assert size_ret[0] == sizeof_device_id, 'No devices found!'
        else:
            if egl:
                gl_ctx_props = Context._properties_list(
                    cl.CL_GL_CONTEXT_KHR,   self._gllib.eglGetCurrentContext(),
                    cl.CL_EGL_DISPLAY_KHR,  self._gllib.eglGetCurrentDisplay(),
                    *platform_props )
            elif sys.platform in ('win32', 'cygwin'):
                gl_ctx_props = Context._properties_list(
                    cl.CL_GL_CONTEXT_KHR,   self._gllib.wglGetCurrentContext(),
                    cl.CL_WGL_HDC_KHR,      self._gllib.wglGetCurrentDC(),
                    *platform_props )
            else:
                gl_ctx_props = Context._properties_list(
                    cl.CL_GL_CONTEXT_KHR,   self._gllib.glXGetCurrentContext(),
                    cl.CL_GLX_DISPLAY_KHR,  self._gllib.glXGetCurrentDisplay(),
                    *platform_props )

            clGetGLContextInfoKHR = platform.get_extension_function("clGetGLContextInfoKHR")

            if clGetGLContextInfoKHR == cl.ffi.NULL:
                raise CLRuntimeError(
                    'platform %s does not provide clGetGLContextInfoKHR!' % platform.name,
                    release=True )

            sizeof_device_id = cl.ffi.sizeof("cl_device_id")
            cl_device_id = cl.ffi.new("cl_device_id *")
            size_ret = cl.ffi.new("size_t *")

            status = clGetGLContextInfoKHR( gl_ctx_props,
                    cl.CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                    sizeof_device_id, cl_device_id, size_ret)

            self.check_error(status, "clGetGLContextInfoKHR", release=True)
            assert size_ret[0] == sizeof_device_id

            self._handle = self._lib.clCreateContext(
                gl_ctx_props, 1, cl_device_id, cl.ffi.NULL, cl.ffi.NULL, err)
            self.check_error(err[0], "clCreateContext", release=True)

        self._devices = [Device(cl_device_id[0])]
        self._platform = self._devices[0].platform
        return self


    def __init__(self, platform, devices=[], properties=[]):
        self._init_empty()
        self._platform = platform
        devices = [ dev if not isinstance(dev, int) else
                          platform.devices[dev] for dev in devices]
        if len(devices) == 0:
            if len(platform.devices) == 1:
                devices = platform.devices
            else:
                raise ValueError('Context can only be created without device list if platform has one device')
        self._devices = devices

        plist = [cl.CL_CONTEXT_PLATFORM, platform.handle] + properties

        n_devices = len(devices)
        device_list = cl.ffi.new("cl_device_id[]", len(devices))
        for i, dev in enumerate(devices):
            device_list[i] = dev.handle

        new_ctx_props = Context._properties_list(*plist)

        err = cl.ffi.new("cl_int *")
        self._handle = self._lib.clCreateContext(
                            new_ctx_props, n_devices, device_list,
                            cl.ffi.NULL, cl.ffi.NULL, err)
        self.check_error(err[0], "clCreateContext", release=True)

    def _add_ref(self, obj):
        self._n_refs += 1

    def _del_ref(self, obj):
        with cl.lock:
            self._n_refs -= 1
            n_refs = self._n_refs
        if n_refs <= 0:
            self._release()


    def get_supported_image_formats(self, flags=cl.CL_MEM_READ_ONLY, image_type=cl.CL_MEM_OBJECT_IMAGE2D):
        n_fmts = cl.ffi.new("cl_uint*")
        status = self._lib.clGetSupportedImageFormats( self.handle,
                    flags, image_type, 0, cl.ffi.NULL, n_fmts )
        self.check_error(status, "clGetSupportedImageFormats")
        fmts = cl.ffi.new("cl_image_format[]", n_fmts[0])
        status = self._lib.clGetSupportedImageFormats( self.handle,
                    flags, image_type, n_fmts[0], fmts, cl.ffi.NULL )
        self.check_error(status, "clGetSupportedImageFormats")
        return fmts

    def create_event_from_gl_sync( self, sync ):
        """Creates an event object linked to an OpenGL sync object.

        Parameters:
            context: A Context created from an OpenGL context or share group.
            sync: The name of a sync object in the associated GL share group.
        Returns:
            Event object.
        """

        try:    # make it fast. If you're using this function you want fast
            clCreateEventFromGLsyncKHR = self._create_event_from_gl_sync
        except AttributeError:
            clCreateEventFromGLsyncKHR = self.platform.get_extension_function('clCreateEventFromGLsyncKHR')
            self._create_event_from_gl_sync = clCreateEventFromGLsyncKHR

        errcode_ret = cl.ffi.new('cl_int*')
        hevent = clCreateEventFromGLsyncKHR(self._handle, sync, errcode_ret )
        self.check_error(errcode_ret[0], 'clCreateEventFromGLsyncKHR')

        event = Event(self, hevent)
        event._host_sync = sync              # host_glbuffer?
        event.from_gl = True
        return event

    def create_queue(self, device=None, flags=0, properties=None):
        """Creates Queue object for the supplied device.

        Parameters:
            device: Device object.
            flags: queue flags (for example
                                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE).
            properties: dictionary of OpenCL 2.0 queue properties.

        Returns:
            Queue object.
        """
        if device is None:
            assert len(self.devices) == 1
            device = self.devices[0]
        return Queue(self, device, flags, properties)


    def create_buffer(self, flags, host_array=None, size=None):
        """Creates Buffer object based on host_array.

        Parameters:
            host_array: numpy array of None.
            size: size if host_array is not a numpy array.

        Returns:
            Buffer object.
        """
        return Buffer(self, flags, host_array, size)


    def create_buffer_from_gl_buffer(self, flags, bufferobj):
        """Creates a Buffer from an OpenGL buffer object.

        Parameters:
            flags: memory access descriptor flags
            renderbuffer: The name of a GL renderbuffer object.
                        The renderbuffer storage must be specified before the
                        image object can be created. The renderbuffer format and
                        dimensions will be used to create the 2D image object.
        Returns:
            Buffer object.
        """
        return Buffer.from_gl_buffer(self, flags, bufferobj)


    def create_image(self, flags, image_format, image_desc, host_array=None):
        """Creates Image object based on host_array.

        Parameters:
            flags: flags supplied for the creation of this image.
            image_format: A 2-tuple of integers.
                        (image_channel_order, image_channel_data_type)
            image_desc: A 10-tuple containing the values of a cl_image_desc struct.
            host_array: host array reference, such as numpy array,
                        will be stored only if flags include CL_MEM_USE_HOST_PTR.

        Returns:
            Image object.
        """
        return Image(self, flags, image_format, image_desc, host_array)


    def create_image_from_gl_renderbuffer(self, flags, renderbuffer):
        """Creates an OpenCL 2D image object from an OpenGL renderbuffer object.

        Parameters:
            flags: memory access descriptor flags
            renderbuffer: The name of a GL renderbuffer object.
                        The renderbuffer storage must be specified before the
                        image object can be created. The renderbuffer format and
                        dimensions will be used to create the 2D image object.
        Returns:
            Image object.
        """

        return Image.from_gl_renderbuffer(self, flags, renderbuffer)


    def create_image_from_gl_texture(self, flags, texture_target, miplevel, texture):
        """Creates a 2D Image from an OpenGL texture object.

        Parameters:
            flags: memory access descriptor flags.
            texture_target: image type of texture.
            miplevel: mipmap level to be used. Usually 0.
            texture: The name of a GL texture object.
                        Can be 1D, 2D, 3D, 1D array, 2D array, cubemap, rectangle
                        or buffer texture object. Must be a complete texture.
        Returns:
            Image object.
        """

        return Image.from_gl_texture(self, flags, texture_target, miplevel, texture)


    def create_program(self, src, include_dirs=(), options="", devices=None,
                       binary=False):
        """Creates and builds OpenCL program from source
           for the supplied devices associated with this context.

        Parameters:
            src: program source.
            include_dirs: list of include directories.
            options: additional build options.
            devices: list of devices on which to build the program
                     (if None will build on all devices).
        Returns:
            Program object.
        """
        return Program(self, self.devices if devices is None else devices,
                       src, include_dirs, options, binary)


    def create_pipe(self, flags, packet_size, max_packets):
        """Creates OpenCL 2.0 pipe.

        Parameters:
            flags: flags for a pipe;
                   as of OpenCL 2.0 only CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY,
                   CL_MEM_READ_WRITE, and CL_MEM_HOST_NO_ACCESS
                   can be specified when creating a pipe object
                   (0 defaults to CL_MEM_READ_WRITE).
            packet_size: size in bytes of a pipe packet
                         (must be greater than 0).
            max_packets: maximum number of packets the pipe can hold
                         (must be greater than 0).
        """
        return Pipe(self, flags, packet_size, max_packets)


    def svm_alloc(self, flags, size, alignment=0):
        """Allocates shared virtual memory (SVM) buffer.

        Parameters:
            flags: flags for a buffer;
                   (CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY,
                    CL_MEM_READ_ONLY, CL_MEM_SVM_FINE_GRAIN_BUFFER,
                    CL_MEM_SVM_ATOMICS).
            size: size in bytes of the SVM buffer to be allocated.
            alignment: the minimum alignment in bytes,
                       it must be a power of two up to the largest
                       data type supported by the OpenCL device,
                       0 defaults to the largest supported alignment.
        """
        return SVM(self, flags, size, alignment)

    def create_sampler(self, normalized_coords, addressing_mode, filter_mode):
        return Sampler(self, normalized_coords, addressing_mode, filter_mode)

    def _release(self):
        if self.handle is not None:
            self._lib.clReleaseContext(self.handle)
            self._handle = None

    def __del__(self):
        self._del_ref(self)


class Device(CL):
    """OpenCL device.

    Attributes:
        platform: Platform object associated with this device.
        type: OpenCL type of the device (integer).
        name: OpenCL name of the device.
        path: opencl4py device identifier,
        version: OpenCL version number of the device (float).
        version_string: OpenCL version string of the device.
        vendor: OpenCL vendor name of the device.
        vendor_id: OpenCL vendor id of the device (integer).
        memsize: global memory size of the device.
        memalign: align in bytes, required for clMapBuffer.
    """

    def __init__(self, handle, platform=None, path="?"):
        super(Device, self).__init__()
        self._handle = handle
        self._platform = platform or Platform(self.platform_id)
        self._path = path # TODO: eventually want to reconstitute this

        self._version_string = self._get_device_info_str(
            cl.CL_DEVICE_OPENCL_C_VERSION)
        n = len("OpenCL C ")
        m = self._version_string.find(" ", n)
        try:
            self._version = float(self._version_string[n:m])
        except ValueError:
            self._version = 0.0

    @property
    def platform(self):
        """
        Platform object associated with this device.
        """
        return self._platform

    @property
    def platform_id(self):
        return self._get_device_info_voidp(cl.CL_DEVICE_PLATFORM)

    @property
    def type(self):
        """
        OpenCL type of the device (integer).
        """
        return self._get_device_info_int(cl.CL_DEVICE_TYPE)

    @property
    def is_cpu(self):
        return self.type == cl.CL_DEVICE_TYPE_CPU

    @property
    def is_gpu(self):
        return self.type == cl.CL_DEVICE_TYPE_GPU

    @property
    def name(self):
        """
        OpenCL name of the device.
        """
        return self._get_device_info_str(cl.CL_DEVICE_NAME)

    @property
    def path(self):
        """
        opencl4py device identifier,
        """
        return self._path

    @property
    def version(self):
        """
        OpenCL version number of the device (float).
        """
        return self._version

    @property
    def version_string(self):
        """
        OpenCL version string of the device.
        """
        return self._version_string

    @property
    def vendor(self):
        """
        OpenCL vendor name of the device.
        """
        return self._get_device_info_str(cl.CL_DEVICE_VENDOR)

    @property
    def vendor_id(self):
        """
        OpenCL vendor id of the device (integer).
        """
        return self._get_device_info_int(cl.CL_DEVICE_VENDOR_ID)

    @property
    def memsize(self):
        """
        Global memory size of the device.
        """
        return self.global_memsize

    @property
    def memalign(self):
        """
        Alignment in bytes, required by clMapBuffer.
        """
        return self.mem_base_addr_align

    @property
    def available(self):
        return self._get_device_info_bool(cl.CL_DEVICE_AVAILABLE)

    @property
    def compiler_available(self):
        return self._get_device_info_bool(cl.CL_DEVICE_COMPILER_AVAILABLE)

    @property
    def little_endian(self):
        return self._get_device_info_bool(cl.CL_DEVICE_ENDIAN_LITTLE)

    @property
    def supports_error_correction(self):
        return self._get_device_info_bool(
            cl.CL_DEVICE_ERROR_CORRECTION_SUPPORT )

    @property
    def host_unified_memory(self):
        return self._get_device_info_bool(cl.CL_DEVICE_HOST_UNIFIED_MEMORY)

    @property
    def supports_images(self):
        return self._get_device_info_bool(cl.CL_DEVICE_IMAGE_SUPPORT)

    @property
    def linker_available(self):
        return self._get_device_info_bool(cl.CL_DEVICE_LINKER_AVAILABLE)

    @property
    def prefers_user_sync(self):
        return self._get_device_info_bool(
            cl.CL_DEVICE_PREFERRED_INTEROP_USER_SYNC)

    @property
    def address_bits(self):
        return self._get_device_info_int(cl.CL_DEVICE_ADDRESS_BITS)

    @property
    def double_fp_config(self):
        return self._get_device_info_int(cl.CL_DEVICE_DOUBLE_FP_CONFIG)

    @property
    def execution_capabilities(self):
        return self._get_device_info_int(cl.CL_DEVICE_EXECUTION_CAPABILITIES)

    @property
    def global_mem_cache_size(self):
        return self._get_device_info_int(cl.CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)

    @property
    def global_mem_cache_line_size(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE)

    @property
    def half_fp_config(self):
        return self._get_device_info_int(cl.CL_DEVICE_HALF_FP_CONFIG)

    @property
    def image2d_max_height(self):
        return self._get_device_info_int(cl.CL_DEVICE_IMAGE2D_MAX_HEIGHT)

    @property
    def image2d_max_width(self):
        return self._get_device_info_int(cl.CL_DEVICE_IMAGE2D_MAX_WIDTH)

    @property
    def image3d_max_depth(self):
        return self._get_device_info_int(cl.CL_DEVICE_IMAGE3D_MAX_DEPTH)

    @property
    def image3d_max_height(self):
        return self._get_device_info_int(cl.CL_DEVICE_IMAGE3D_MAX_HEIGHT)

    @property
    def image3d_max_width(self):
        return self._get_device_info_int(cl.CL_DEVICE_IMAGE3D_MAX_WIDTH)

    @property
    def image_max_buffer_size(self):
        return self._get_device_info_int(cl.CL_DEVICE_IMAGE_MAX_BUFFER_SIZE)

    @property
    def image_max_array_size(self):
        return self._get_device_info_int(cl.CL_DEVICE_IMAGE_MAX_ARRAY_SIZE)

    @property
    def local_memsize(self):
        return self._get_device_info_int(cl.CL_DEVICE_LOCAL_MEM_SIZE)

    @property
    def global_memsize(self):
        return self._get_device_info_int(cl.CL_DEVICE_GLOBAL_MEM_SIZE)

    @property
    def max_clock_frequency(self):
        return self._get_device_info_int(cl.CL_DEVICE_MAX_CLOCK_FREQUENCY)

    @property
    def max_compute_units(self):
        return self._get_device_info_int(cl.CL_DEVICE_MAX_COMPUTE_UNITS)

    @property
    def max_constant_args(self):
        return self._get_device_info_int(cl.CL_DEVICE_MAX_CONSTANT_ARGS)

    @property
    def max_constant_buffer_size(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)

    @property
    def max_mem_alloc_size(self):
        return self._get_device_info_int(cl.CL_DEVICE_MAX_MEM_ALLOC_SIZE)

    @property
    def max_parameter_size(self):
        return self._get_device_info_int(cl.CL_DEVICE_MAX_PARAMETER_SIZE)

    @property
    def max_read_image_args(self):
        return self._get_device_info_int(cl.CL_DEVICE_MAX_READ_IMAGE_ARGS)

    @property
    def max_work_group_size(self):
        return self._get_device_info_int(cl.CL_DEVICE_MAX_WORK_GROUP_SIZE)

    @property
    def max_work_item_dimensions(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)

    @property
    def max_write_image_args(self):
        return self._get_device_info_int(cl.CL_DEVICE_MAX_WRITE_IMAGE_ARGS)

    @property
    def mem_base_addr_align(self):
        return self._get_device_info_int(cl.CL_DEVICE_MEM_BASE_ADDR_ALIGN)

    @property
    def min_data_type_align_size(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE)

    @property
    def preferred_vector_width_char(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR)

    @property
    def preferred_vector_width_short(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT)

    @property
    def preferred_vector_width_int(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT)

    @property
    def preferred_vector_width_long(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG)

    @property
    def preferred_vector_width_float(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)

    @property
    def preferred_vector_width_double(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)

    @property
    def preferred_vector_width_half(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF)

    @property
    def printf_buffer_size(self):
        return self._get_device_info_int(cl.CL_DEVICE_PRINTF_BUFFER_SIZE)

    @property
    def profiling_timer_resolution(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PROFILING_TIMER_RESOLUTION)

    @property
    def reference_count(self):
        return self._get_device_info_int(cl.CL_DEVICE_REFERENCE_COUNT)

    @property
    def single_fp_config(self):
        return self._get_device_info_int(cl.CL_DEVICE_SINGLE_FP_CONFIG)

    @property
    def built_in_kernels(self):
        return [kernel.strip() for kernel in self._get_device_info_str(
            cl.CL_DEVICE_BUILT_IN_KERNELS).split(';')
            if kernel.strip()]

    @property
    def extensions(self):
        return [ext.strip() for ext in self._get_device_info_str(
            cl.CL_DEVICE_EXTENSIONS).split(' ')
            if ext.strip()]

    @property
    def profile(self):
        return self._get_device_info_str(cl.CL_DEVICE_PROFILE)

    @property
    def driver_version(self):
        return self._get_device_info_str(cl.CL_DRIVER_VERSION)

    @property
    def max_work_item_sizes(self):
        value = cl.ffi.new("size_t[]", self.max_work_item_dimensions)
        err = self._lib.clGetDeviceInfo(
                    self._handle, cl.CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    cl.ffi.sizeof(value), value, cl.ffi.NULL)
        if err:
            return None
        return list(value)

    @property
    def pipe_max_packet_size(self):
        return self._get_device_info_int(cl.CL_DEVICE_PIPE_MAX_PACKET_SIZE)

    @property
    def pipe_max_active_reservations(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS)

    @property
    def svm_capabilities(self):
        return self._get_device_info_int(cl.CL_DEVICE_SVM_CAPABILITIES)

    @property
    def preferred_platform_atomic_alignment(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT)

    @property
    def preferred_global_atomic_alignment(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT)

    @property
    def preferred_local_atomic_alignment(self):
        return self._get_device_info_int(
            cl.CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT)

    def _get_device_info_bool(self, name):
        value = cl.ffi.new("cl_bool[]", 1)
        err = self._lib.clGetDeviceInfo(
                self._handle, name, cl.ffi.sizeof(value), value, cl.ffi.NULL)
        self.check_error(err, "clGetDeviceInfo")
        return bool(value[0])

    def _get_device_info_int(self, name):
        value = cl.ffi.new("uint64_t[]", 1)
        err = self._lib.clGetDeviceInfo(
                self._handle, name, cl.ffi.sizeof(value), value, cl.ffi.NULL)
        self.check_error(err, "clGetDeviceInfo")
        return int(value[0])

    def _get_device_info_str(self, name):
        value = cl.ffi.new("char[]", 1024)
        err = self._lib.clGetDeviceInfo(
                self._handle, name, cl.ffi.sizeof(value), value, cl.ffi.NULL)
        self.check_error(err, "clGetDeviceInfo")
        return cl.ffi.string(value).decode("utf-8")

    def _get_device_info_voidp(self, name):
        value = cl.ffi.new("void**")
        err = self._lib.clGetDeviceInfo(
                self._handle, name, cl.ffi.sizeof(value), value, cl.ffi.NULL)
        self.check_error(err, "clGetDeviceInfo")
        return value[0]

    def __repr__(self):
        return '<opencl4py.Device %r>' % self.name


class Platform(CL):
    """OpenCL platform.

    Attributes:
        devices: list of Device objects available on this platform.
        name: OpenCL name of the platform.
        path: opencl4py platform identifier.
    """
    def __init__(self, handle, path="?"):
        super(Platform, self).__init__()
        self._handle = handle
        self._path = path
        self._name = None
        self._devices = None
        self._extensions = None
        self._extension_functions = {}

    def _get_platform_info(self, name, value):
        sz = cl.ffi.new('size_t*')
        err = self._lib.clGetPlatformInfo(
            self._handle, name, cl.ffi.sizeof(value), value, sz)
        self.check_error(err, "clGetPlatformInfo")
        return sz


    def _get_platform_info_str(self, name):
        value = cl.ffi.new("char[]", 1024)
        self._get_platform_info(name, value)
        return cl.ffi.string(value).decode("utf-8")


    @property
    def devices(self):
        """
        List of Device objects available on this platform.
        """
        if not self._devices:
            nn = cl.ffi.new("cl_uint[]", 1)
            n = self._lib.clGetDeviceIDs(self.handle, cl.CL_DEVICE_TYPE_ALL,
                                         0, cl.ffi.NULL, nn)
            self.check_error(n, 'clGetDeviceIDs')

            ids = cl.ffi.new("cl_device_id[]", nn[0])
            n = self._lib.clGetDeviceIDs(self.handle, cl.CL_DEVICE_TYPE_ALL,
                                         nn[0], ids, nn)
            self.check_error(n, 'clGetDeviceIDs')

            self._devices = list(Device(dev_id, self,
                                        "%s:%d" % (self.path, dev_num))
                                 for dev_num, dev_id in enumerate(ids))
        return self._devices

    @property
    def name(self):
        """
        OpenCL name of the platform.
        """
        if not self._name:
            self._name = self._get_platform_info_str(cl.CL_PLATFORM_NAME)
        return self._name

    @property
    def path(self):
        """
        opencl4py platform identifier.
        """
        return self._path

    @property
    def extensions(self):
        if not self._extensions:
            self._extensions = [ ext.strip() for ext in self._get_platform_info_str(
                cl.CL_PLATFORM_EXTENSIONS).split(' ')
                if ext.strip()]
        return self._extensions

    def get_extension_function_address(self, fn_name):
        return self._lib.clGetExtensionFunctionAddressForPlatform(self.handle, fn_name)

    def get_extension_function(self, fn_name):
        try:
            return self._extension_functions[fn_name]
        except KeyError:
            fn_addr = self.get_extension_function_address( fn_name )
            fn_type = typeof_function( cl.ffi, fn_name )
            fn = cl.ffi.cast(fn_type, fn_addr)
            self._extension_functions[fn_name] = fn
            return fn

    def __iter__(self):
        return iter(self.devices)


    def create_context(self, devices=[]):
        """Creates OpenCL context on this platform and selected devices.

        Parameters:
            devices: list of Device objects.

        Returns:
            Context object.
        """
        return Context(self, devices)

    def __repr__(self):
        return '<opencl4py.Platform %r>' % self.name


class Platforms(CL):
    """List of OpenCL plaforms.

    Attributes:
        platforms: list of Platform objects.
    """
    def __init__(self):
        cl.initialize()
        super(Platforms, self).__init__()
        ids = Platforms.get_ids()
        self._platforms = list(Platform(p_id, str(p_num))
                               for p_num, p_id in enumerate(ids))

    @property
    def platforms(self):
        return self._platforms

    def __iter__(self):
        return iter(self.platforms)

    def __getitem__(self, what):
        return self.platforms.__getitem__(what)

    @staticmethod
    def get_ids():
        nn = cl.ffi.new("cl_uint[]", 1)
        n = cl.lib.clGetPlatformIDs(0, cl.ffi.NULL, nn)
        check_error(n, 'clGetPlatformIDs')

        ids = cl.ffi.new("cl_platform_id[]", nn[0])
        n = cl.lib.clGetPlatformIDs(nn[0], ids, nn)
        check_error(n, 'clGetPlatformIDs')
        return ids

    def dump_devices(self):
        """Returns string with information about OpenCL platforms and devices.
        """
        if not len(self.platforms):
            return "No OpenCL devices available."
        lines = []
        for i, platform in enumerate(self.platforms):
            lines.append("Platform %d: %s" % (i, platform.name.strip()))
            for j, device in enumerate(platform.devices):
                lines.append("\tDevice %d: %s (%d Mb, %d align, %s)" % (
                    j, device.name.strip(), device.memsize // (1024 * 1024),
                    device.memalign, device.version_string.strip()))
        return "\n".join(lines)

    def create_context_from_string(self, ctx):
        assert isinstance(ctx, basestring)
        idx = ctx.find(":")
        if idx >= 0:
            try:
                platform_number = int(ctx[:idx]) if len(ctx[:idx]) else 0
            except ValueError:
                raise ValueError("Incorrect platform number")
            ctx = ctx[idx + 1:]
        else:
            platform_number = 0
        device_strings = ctx.split(",")
        device_numbers = []
        try:
            for s in device_strings:
                device_numbers.append(int(s) if len(s) else 0)
        except ValueError:
            raise ValueError("Incorrect device number")
        try:
            platform = self.platforms[platform_number]
        except IndexError:
            raise IndexError("Platform index is out of range")
        devices = []
        try:
            for i in device_numbers:
                devices.append(platform.devices[i])
        except IndexError:
            raise IndexError("Device index is out of range")
        return platform.create_context(devices)



    def create_some_context(self):
        """Returns Context object with some OpenCL platform, devices attached.

        If environment variable PYOPENCL_CTX is set and not empty,
        gets context based on it, format is:
        <platform number>:<comma separated device numbers>
        (Examples: 0:0 - first platform, first device,
                   1:0,2 - second platform, first and third devices).

        If PYOPENCL_CTX is not set and os.isatty(0) == True, then
        displays available devices and reads line from stdin in the same
        format as PYOPENCL_CTX.

        Else chooses first platform and device.
        """
        if len(self.platforms) == 1 and len(self.platforms[0].devices) == 1:
            return self.platforms[0].create_context(self.platforms[0].devices)
        import os
        ctx = os.environ.get("PYOPENCL_CTX")
        if ctx is None or not len(ctx):
            if os.isatty(0):
                import sys
                sys.stdout.write(
                    "\nEnter "
                    "<platform number>:<comma separated device numbers> or "
                    "set PYOPENCL_CTX environment variable.\n"
                    "Examples: 0:0 - first platform, first device;\n"
                    "          1:0,2 - second platform, first and third "
                    "devices.\n"
                    "\nOpenCL devices available:\n\n%s\n\n" %
                    (self.dump_devices()))
                sys.stdout.flush()
                ctx = sys.stdin.readline().strip()
            else:
                ctx = ""
        return self.create_context_from_string(ctx)
