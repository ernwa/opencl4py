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
OpenCL cffi bindings.
"""
import cffi
import threading
from _cl_defines import *
#: ffi parser
ffi = None
#: Loaded shared library
lib = None
gllib = None
#: Lock
lock = threading.Lock()

clsrc = """
typedef int32_t             cl_int;
typedef uint32_t            cl_uint;
typedef uint64_t            cl_ulong;
typedef cl_uint             cl_bool;                     /* WARNING!  Unlike cl_ types in cl_platform.h, cl_bool is not guaranteed to be the same size as the bool in kernels. */
typedef cl_ulong            cl_bitfield;
typedef cl_bitfield         cl_device_type;
typedef cl_uint             cl_platform_info;
typedef cl_uint             cl_device_info;
typedef cl_bitfield         cl_device_fp_config;
typedef cl_uint             cl_device_mem_cache_type;
typedef cl_uint             cl_device_local_mem_type;
typedef cl_bitfield         cl_device_exec_capabilities;
typedef cl_bitfield         cl_device_svm_capabilities;
typedef cl_bitfield         cl_command_queue_properties;
typedef intptr_t            cl_device_partition_property;
typedef cl_bitfield         cl_device_affinity_domain;

typedef intptr_t            cl_context_properties;
typedef cl_uint             cl_context_info;
typedef cl_bitfield         cl_queue_properties;
typedef cl_uint             cl_command_queue_info;
typedef cl_uint             cl_channel_order;
typedef cl_uint             cl_channel_type;
typedef cl_bitfield         cl_mem_flags;
typedef cl_bitfield         cl_svm_mem_flags;
typedef cl_uint             cl_mem_object_type;
typedef cl_uint             cl_mem_info;
typedef cl_bitfield         cl_mem_migration_flags;
typedef cl_uint             cl_image_info;
typedef cl_uint             cl_buffer_create_type;
typedef cl_uint             cl_addressing_mode;
typedef cl_uint             cl_filter_mode;
typedef cl_uint             cl_sampler_info;
typedef cl_bitfield         cl_map_flags;
typedef intptr_t            cl_pipe_properties;
typedef cl_uint             cl_pipe_info;
typedef cl_uint             cl_program_info;
typedef cl_uint             cl_program_build_info;
typedef cl_uint             cl_program_binary_type;
typedef cl_int              cl_build_status;
typedef cl_uint             cl_kernel_info;
typedef cl_uint             cl_kernel_arg_info;
typedef cl_uint             cl_kernel_arg_address_qualifier;
typedef cl_uint             cl_kernel_arg_access_qualifier;
typedef cl_bitfield         cl_kernel_arg_type_qualifier;
typedef cl_uint             cl_kernel_work_group_info;
typedef cl_uint             cl_kernel_sub_group_info;
typedef cl_uint             cl_event_info;
typedef cl_uint             cl_command_type;
typedef cl_uint             cl_profiling_info;
typedef cl_bitfield         cl_sampler_properties;
typedef cl_uint             cl_kernel_exec_info;

typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_command_queue;
typedef void* cl_mem;
typedef void* cl_event;
typedef void* cl_sampler;

typedef struct _cl_image_format {
    cl_uint        image_channel_order;
    cl_uint        image_channel_data_type;
} cl_image_format;

typedef struct _cl_image_desc {
    cl_mem_object_type      image_type;
    size_t                  image_width;
    size_t                  image_height;
    size_t                  image_depth;
    size_t                  image_array_size;
    size_t                  image_row_pitch;
    size_t                  image_slice_pitch;
    cl_uint                 num_mip_levels;
    cl_uint                 num_samples;
    union {                                         // fixme?
        cl_mem                  buffer;
        cl_mem                  mem_object;
    };
} cl_image_desc;

typedef struct __GLsync *cl_GLsync;

cl_int clGetPlatformIDs(cl_uint num_entries,
                        cl_platform_id *platforms,
                        cl_uint *num_platforms);

cl_int clGetDeviceIDs(cl_platform_id  platform,
                      cl_device_type device_type,
                      cl_uint num_entries,
                      cl_device_id *devices,
                      cl_uint *num_devices);

cl_int clGetPlatformInfo(cl_platform_id platform,
                         cl_platform_info param_name,
                         size_t param_value_size,
                         void *param_value,
                         size_t *param_value_size_ret);

cl_int clGetDeviceInfo(cl_device_id device,
                       cl_device_info param_name,
                       size_t param_value_size,
                       void *param_value,
                       size_t *param_value_size_ret);

cl_context clCreateContext(const cl_context_properties *properties,
                           cl_uint num_devices,
                           const cl_device_id *devices,
                           void *p_notify,
                           void *user_data,
                           cl_int *errcode_ret);

cl_int clReleaseContext(cl_context context);

cl_program clCreateProgramWithSource(cl_context context,
                                     cl_uint count,
                                     const char **strings,
                                     const size_t *lengths,
                                     cl_int *errcode_ret);

cl_program clCreateProgramWithBinary(cl_context context,
                                     cl_uint num_devices,
                                     const cl_device_id *device_list,
                                     const size_t *lengths,
                                     const unsigned char **binaries,
                                     cl_int *binary_status,
                                     cl_int *errcode_ret);

cl_int clReleaseProgram(cl_program program);

cl_int clBuildProgram(cl_program program,
                      cl_uint num_devices,
                      const cl_device_id *device_list,
                      const char *options,
                      void *p_notify,
                      void *user_data);

cl_int clGetProgramBuildInfo(cl_program program,
                             cl_device_id device,
                             cl_program_build_info param_name,
                             size_t param_value_size,
                             void *param_value,
                             size_t *param_value_size_ret);

cl_int clGetProgramInfo(cl_program program,
                        cl_program_info param_name,
                        size_t param_value_size,
                        void *param_value,
                        size_t *param_value_size_ret);

cl_kernel clCreateKernel(cl_program program,
                         const char *kernel_name,
                         cl_int *errcode_ret);

cl_int clReleaseKernel(cl_kernel kernel);

cl_int clGetKernelInfo(cl_kernel kernel,
                       cl_kernel_info param_name,
                       size_t param_value_size,
                       void *param_value,
                       size_t *param_value_size_ret);

cl_int clGetKernelWorkGroupInfo(cl_kernel kernel,
                                cl_device_id device,
                                cl_kernel_work_group_info param_name,
                                size_t param_value_size,
                                void *param_value,
                                size_t *param_value_size_ret);

cl_int clSetKernelArg(cl_kernel kernel,
                      cl_uint arg_index,
                      size_t arg_size,
                      const void *arg_value);

cl_command_queue clCreateCommandQueue(
                                cl_context context,
                                cl_device_id device,
                                cl_command_queue_properties properties,
                                cl_int *errcode_ret);

cl_command_queue clCreateCommandQueueWithProperties(
                                cl_context context,
                                cl_device_id device,
                                const cl_queue_properties *properties,
                                cl_int *errcode_ret);

cl_int clReleaseCommandQueue(cl_command_queue command_queue);


cl_int clGetMemObjectInfo(cl_mem   memobj,
               cl_mem_info         param_name,
               size_t              param_value_size,
               void *              param_value,
               size_t *            param_value_size_ret);

cl_mem clCreateBuffer(cl_context context,
                      cl_mem_flags flags,
                      size_t size,
                      void *host_ptr,
                      cl_int *errcode_ret);

cl_mem clCreateSubBuffer(cl_mem buffer,
                         cl_mem_flags flags,
                         cl_buffer_create_type buffer_create_type,
                         const void *buffer_create_info,
                         cl_int *errcode_ret);

cl_int clReleaseMemObject(cl_mem memobj);

void* clEnqueueMapBuffer(cl_command_queue command_queue,
                         cl_mem buffer,
                         cl_bool blocking_map,
                         cl_map_flags map_flags,
                         size_t offset,
                         size_t size,
                         cl_uint num_events_in_wait_list,
                         const cl_event *event_wait_list,
                         cl_event *event,
                         cl_int *errcode_ret);

cl_int clEnqueueUnmapMemObject(cl_command_queue command_queue,
                               cl_mem memobj,
                               void *mapped_ptr,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event);

cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
                           cl_mem buffer,
                           cl_bool blocking_read,
                           size_t offset,
                           size_t size,
                           void *ptr,
                           cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list,
                           cl_event *event);

cl_int clEnqueueReadBufferRect(cl_command_queue command_queue,
                        cl_mem               buffer ,
                        cl_bool              blocking_read ,
                        const size_t *       buffer_offset ,
                        const size_t *       host_offset ,
                        const size_t *       region ,
                        size_t               buffer_row_pitch ,
                        size_t               buffer_slice_pitch ,
                        size_t               host_row_pitch ,
                        size_t               host_slice_pitch ,
                        void *               ptr ,
                        cl_uint              num_events_in_wait_list ,
                        const cl_event *     event_wait_list ,
                        cl_event *           event );

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
                            cl_mem buffer,
                            cl_bool blocking_write,
                            size_t offset,
                            size_t size,
                            const void *ptr,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event);

cl_int clEnqueueCopyBuffer(cl_command_queue command_queue,
                           cl_mem src_buffer,
                           cl_mem dst_buffer,
                           size_t src_offset,
                           size_t dst_offset,
                           size_t size,
                           cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list,
                           cl_event *event);

cl_int clEnqueueCopyBufferRect(cl_command_queue command_queue,
                               cl_mem src_buffer,
                               cl_mem dst_buffer,
                               const size_t *src_origin,
                               const size_t *dst_origin,
                               const size_t *region,
                               size_t src_row_pitch,
                               size_t src_slice_pitch,
                               size_t dst_row_pitch,
                               size_t dst_slice_pitch,
                               cl_uint num_events_in_wait_list,
                               const cl_event *event_wait_list,
                               cl_event *event);

cl_int clEnqueueFillBuffer(cl_command_queue command_queue,
                           cl_mem buffer,
                           const void *pattern,
                           size_t pattern_size,
                           size_t offset,
                           size_t size,
                           cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list,
                           cl_event *event);


cl_int clEnqueueMigrateMemObjects (	cl_command_queue  command_queue ,
                         	cl_uint                   num_mem_objects ,
                         	const cl_mem              *mem_objects ,
                         	cl_mem_migration_flags    flags ,
                         	cl_uint                   num_events_in_wait_list ,
                         	const cl_event            *event_wait_list ,
                         	cl_event                  *event );

cl_int clWaitForEvents(     cl_uint                   num_events,
                            const cl_event            *event_list);

cl_int clReleaseEvent(cl_event event);


cl_int clEnqueueMarker (    cl_command_queue          command_queue,
 	                        cl_event *                event );

cl_int clEnqueueBarrier(    cl_command_queue        command_queue);

cl_int clFlush(             cl_command_queue        command_queue);

cl_int clFinish(            cl_command_queue        command_queue);

cl_int clEnqueueNDRangeKernel(cl_command_queue        command_queue,
                              cl_kernel               kernel,
                              cl_uint                 work_dim,
                              const size_t *          global_work_offset,
                              const size_t *          global_work_size,
                              const size_t *          local_work_size,
                              cl_uint                 num_events_in_wait_list,
                              const cl_event *        event_wait_list,
                              cl_event *              event );

cl_int clGetEventProfilingInfo(cl_event               event,
                               cl_profiling_info      param_name,
                               size_t                 param_value_size,
                               void *                 param_value,
                               size_t *               param_value_size_ret );

cl_mem clCreatePipe(           cl_context             context,
                                cl_mem_flags          flags,
                                cl_uint               pipe_packet_size,
                                cl_uint               pipe_max_packets,
                                const cl_pipe_properties * properties,
                                cl_int *              errcode_ret );

cl_int clGetPipeInfo(cl_mem    pipe,
              cl_pipe_info     param_name,
              size_t           param_value_size,
              void *           param_value,
              size_t *         param_value_size_ret);


void *clSVMAlloc(cl_context context,
                 cl_svm_mem_flags flags,
                 size_t size,
                 unsigned int alignment);

void clSVMFree(cl_context context,
               void *svm_pointer);

cl_int clEnqueueSVMMap(cl_command_queue command_queue,
                       cl_bool blocking_map,
                       cl_map_flags map_flags,
                       void *svm_ptr,
                       size_t size,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list,
                       cl_event *event);

cl_int clEnqueueSVMUnmap(
                cl_command_queue command_queue,
                void *svm_ptr,
                cl_uint  num_events_in_wait_list,
                const cl_event *event_wait_list,
                cl_event *event);

cl_int clSetKernelArgSVMPointer(
                cl_kernel kernel,
                cl_uint arg_index,
                const void *arg_value);

cl_int clEnqueueSVMMemcpy(
                cl_command_queue command_queue,
                cl_bool blocking_copy,
                void *dst_ptr,
                const void *src_ptr,
                size_t size,
                cl_uint num_events_in_wait_list,
                const cl_event *event_wait_list,
                cl_event *event);

cl_int clEnqueueSVMMemFill(cl_command_queue command_queue,
                void *svm_ptr,
                const void *pattern,
                size_t pattern_size,
                size_t size,
                cl_uint num_events_in_wait_list,
                const cl_event *event_wait_list,
                cl_event *event);



cl_sampler clCreateSampler(cl_context context,
                cl_bool normalized_coords,
                cl_addressing_mode addressing_mode,
                cl_filter_mode filter_mode,
                cl_int * errcode_ret); // deprecated CL1.2

cl_sampler clCreateSamplerWithProperties(cl_context context,
                const cl_sampler_properties * normalized_coords,
                cl_int * errcode_ret);

cl_int clRetainSampler(cl_sampler sampler);

cl_int clReleaseSampler(cl_sampler sampler);

cl_int clGetSamplerInfo(cl_sampler sampler,
                 cl_sampler_info param_name,
                 size_t param_value_size,
                 void * param_value,
                 size_t * param_value_size_ret);

cl_int clGetSupportedImageFormats(cl_context context,
                cl_mem_flags flags,
                cl_mem_object_type image_type,
                cl_uint num_entries,
                cl_image_format * image_formats,
                cl_uint * num_image_formats);

cl_int clGetImageInfo(cl_mem image,
           cl_image_info    param_name,
           size_t           param_value_size,
           void *           param_value,
           size_t *         param_value_size_ret);


cl_mem clCreateImage(cl_context context,
              cl_mem_flags flags,
              const cl_image_format * image_format,
              const cl_image_desc * image_desc,
              void * host_ptr,
              cl_int * errcode_ret);

void * clEnqueueMapImage(cl_command_queue command_queue,
              cl_mem image,
              cl_bool blocking_map,
              cl_map_flags map_flags,
              const size_t * origin /* [3] */,
              const size_t * region /* [3] */,
              size_t * image_row_pitch,
              size_t * image_slice_pitch,
              cl_uint num_events_in_wait_list,
              const cl_event * event_wait_list,
              cl_event * event,
              cl_int * errcode_ret);

cl_int clEnqueueReadImage(cl_command_queue command_queue,
               cl_mem image,
               cl_bool blocking_read,
               const size_t * origin /* [3] */,
               const size_t * region /* [3] */,
               size_t row_pitch,
               size_t slice_pitch,
               void * ptr,
               cl_uint num_events_in_wait_list,
               const cl_event * event_wait_list,
               cl_event * event);


cl_int clEnqueueWriteImage(cl_command_queue command_queue,
                cl_mem image,
                cl_bool blocking_write,
                const size_t * origin /* [3] */,
                const size_t * region /* [3] */,
                size_t input_row_pitch,
                size_t input_slice_pitch,
                const void * ptr,
                cl_uint num_events_in_wait_list,
                const cl_event * event_wait_list,
                cl_event * event);

cl_int clEnqueueFillImage(cl_command_queue command_queue,
               cl_mem image,
               const void * fill_color,
               const size_t * origin /* [3] */,
               const size_t * region /* [3] */,
               cl_uint num_events_in_wait_list,
               const cl_event * event_wait_list,
               cl_event * event);

cl_int clEnqueueCopyImage(cl_command_queue command_queue,
               cl_mem src_image,
               cl_mem dst_image,
               const size_t * src_origin  /* [3] */,
               const size_t * dst_origin  /* [3] */,
               const size_t * region  /* [3] */,
               cl_uint num_events_in_wait_list,
               const cl_event * event_wait_list,
               cl_event * event);

cl_int clEnqueueCopyImageToBuffer(cl_command_queue command_queue,
               cl_mem src_image,
               cl_mem dst_buffer,
               const size_t * origin /* [3] */,
               const size_t * region /* [3] */,
               size_t dst_offset,
               cl_uint num_events_in_wait_list,
               const cl_event * event_wait_list,
               cl_event * event);

cl_int clEnqueueCopyBufferToImage(
                   cl_command_queue         command_queue,
                   cl_mem                   src_buffer,
                   cl_mem                   dst_image,
                   size_t                   src_offset,
                   const size_t *           origin /* [3] */,
                   const size_t *           region /* [3] */,
                   cl_uint                  num_events_in_wait_list,
                   const cl_event *         event_wait_list,
                   cl_event *               event);

/* OPENGL INTEROP! */
    typedef int          cl_GLint;
    typedef unsigned int cl_GLenum;
    typedef unsigned int cl_GLuint;
    typedef cl_uint     cl_gl_object_type;
    typedef cl_uint     cl_gl_texture_info;
    typedef cl_uint     cl_gl_platform_info;
    typedef cl_uint     cl_gl_context_info;


cl_int	clGetGLContextInfoAPPLE(
                    cl_context             context,
					void *                 platform_gl_ctx,
					cl_gl_platform_info    param_name,
					size_t                 param_value_size,
					void *                 param_value,
					size_t *               param_value_size_ret);

cl_mem clCreateFromGLBuffer(
                    cl_context              context,
                    cl_mem_flags            flags,
                    cl_GLuint               bufobj,
                    cl_int *                errcode_ret);

cl_mem clCreateFromGLTexture(
                    cl_context              context,
                    cl_mem_flags            flags,
                    cl_GLenum               target,
                    cl_GLint                miplevel,
                    cl_GLuint               texture,
                    cl_int *                errcode_ret);

cl_mem clCreateFromGLRenderbuffer(
                    cl_context              context,
                    cl_mem_flags            flags,
                    cl_GLuint               renderbuffer,
                    cl_int *                errcode_ret);

cl_int clGetGLObjectInfo(
                    cl_mem                memobj,
                    cl_gl_object_type *   gl_object_type,
                    cl_GLuint *           gl_object_name);

cl_int clGetGLTextureInfo(
                    cl_mem               memobj,
                    cl_gl_texture_info   param_name,
                    size_t               param_value_size,
                    void *               param_value,
                    size_t *             param_value_size_ret);

cl_int clEnqueueAcquireGLObjects(
                    cl_command_queue      command_queue,
                    cl_uint               num_objects,
                    const cl_mem *        mem_objects,
                    cl_uint               num_events_in_wait_list,
                    const cl_event *      event_wait_list,
                    cl_event *            event);

cl_int clEnqueueReleaseGLObjects(
                    cl_command_queue      command_queue,
                    cl_uint               num_objects,
                    const cl_mem *        mem_objects,
                    cl_uint               num_events_in_wait_list,
                    const cl_event *      event_wait_list,
                    cl_event *            event);


/* Deprecated OpenCL 1.1 APIs */
cl_mem clCreateFromGLTexture2D(cl_context      context,
                        cl_mem_flags    flags,
                        cl_GLenum       target,
                        cl_GLint        miplevel,
                        cl_GLuint       texture,
                        cl_int *        errcode_ret);

cl_mem clCreateFromGLTexture3D(cl_context      context,
                        cl_mem_flags    flags,
                        cl_GLenum       target,
                        cl_GLint        miplevel,
                        cl_GLuint       texture,
                        cl_int *        errcode_ret);

cl_event clCreateEventFromGLsyncKHR(
                           cl_context context,
                           cl_GLsync cl_GLsync,
                           cl_int errcode_ret);

// Mac (darwin) CGL functions

typedef void * CGLContextObj;
typedef void * CGLShareGroupObj;
CGLContextObj CGLGetCurrentContext(void);
CGLShareGroupObj CGLGetShareGroup(CGLContextObj ctx);

// Windows WGL functions

void * wglGetCurrentDC(void);
void * wglGetCurrentContext(void);

// x windows GLX functions

void * glXGetCurrentContext( void );
void * glXGetCurrentDisplay( void );


void * clGetExtensionFunctionAddressForPlatform(
    cl_platform_id platform,
 	const char *funcname );


// extension functions and function pointers


// typedef cl_int (*clGetGLContextInfoKHR_fn)(
//         cl_context_properties *, cl_gl_context_info, size_t, void *, size_t *);

"""

extensions = {
    'cl_khr_gl_sharing': """
            cl_int  clGetGLContextInfoKHR(
                        cl_context_properties *     properties,
                        cl_gl_context_info          param_name,
                        size_t                      param_value_size,
                        void *                      param_value,
                        size_t *                    param_value_size_ret);""",

    'cl_khr_icd':        """
            cl_int  clIcdGetPlatformIDsKHR(
                        cl_uint                     num_entries,
                        cl_platform_id *            platforms,
                        cl_uint *                   num_platforms);""",

    'cl_ext_migrate_memobject': """
            typedef     cl_bitfield                 cl_mem_migration_flags_ext;
            cl_int  clEnqueueMigrateMemObjectEXT(
                        cl_command_queue            command_queue,
                        cl_uint                     num_mem_objects,
                        const cl_mem *              mem_objects,
                        cl_mem_migration_flags_ext flags,
                        cl_uint                     num_events_in_wait_list,
                        const cl_event *            event_wait_list,
                        cl_event *                  event);""",

    'cl_ext_device_fission': """
            typedef     cl_bitfield                 cl_device_partition_property_ext;
            cl_int  clReleaseDeviceEXT( cl_device_id device );

            cl_int  clRetainDeviceEXT( cl_device_id  device );

            cl_int  clCreateSubDevicesEXT(
                        cl_device_id                in_device,
                        const cl_device_partition_property_ext * properties,
                        cl_uint                     num_entries,
                        cl_device_id *              out_devices,
                        cl_uint *                   num_devices );""",

}


def initialize(
        backends=("libOpenCL.so", "OpenCL.dll", "OpenCL"),
        gl_backends=("libGL.so", "OpenGL32.dll", "OpenGL")):    # FIXME: add EGL

    global lib, ffi, gllib, extensions, lock

    with lock:
        if ffi:
            print( '<opencl4py> reinitializing ffi. Things may not work right.' )
        ffi = cffi.FFI()
        ffi.cdef(clsrc)

        if lib is None:
            for fnlib in backends:
                try:
                    lib = ffi.dlopen(fnlib)
                    break
                except OSError:
                    pass
            else:
                raise OSError("Could not load OpenCL library")

        if gllib is None:
            for fnlib in gl_backends:
                try:
                    gllib = ffi.dlopen(fnlib)
                    break
                except OSError:
                    pass

        ffi.cdef('\n'.join(extensions.values()))    # register the known extensions
